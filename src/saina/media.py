"""Image and audio inputs: conversion to data URLs, validation, decoding, and the audio spectrogram.

A request carries ``images`` and ``audios`` as base64 data URLs. Clients convert local paths, http(s)
URLs, bytes, PIL images and ``(samples, sample_rate)`` arrays to data URLs (:func:`image_data_url`,
:func:`audio_data_url`) with the standard library only. A server validates them (:func:`check_media`)
and decodes them (:func:`decode_media`) into the images the model reads: the request's images in order,
then one log-mel spectrogram image per audio clip (:func:`mel_spectrogram_png`).

Decoding needs ``pip install 'saina[media]'`` (Pillow, NumPy, SciPy, soundfile); importing this module
does not.
"""
import base64
import binascii
import io
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache

MAX_IMAGES = 8
MAX_AUDIOS = 4
MAX_MEDIA_BYTES = 10_000_000  # per image or audio clip, decoded
IMAGE_TYPES = ('png', 'jpeg', 'webp')
AUDIO_TYPES = ('wav', 'x-wav', 'flac', 'mpeg', 'ogg', 'webm')
# Every image is read at up to 1.6 megapixels and at least 65,536 pixels, keeping its aspect ratio.
IMAGE_MIN_PIXELS = 65_536
IMAGE_MAX_PIXELS = 1_638_400
MAX_IMAGE_SOURCE_PIXELS = 40_000_000  # larger images are refused before decoding
MAX_ASPECT_RATIO = 200

# Audio is read as mono 16 kHz; a clip may be at most 10 s long (longer clips are refused, never cut).
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 10.0
MIN_AUDIO_SECONDS = 1.0  # shorter clips are padded with silence at the end
N_FFT = 1024
HOP = 160  # 10 ms: one spectrogram column per hop, 100 columns per second
N_MELS = 128
F_MIN, F_MAX = 0.0, 8_000.0
TOP_DB = 80.0

DOWNLOAD_TIMEOUT = 30
INPUT_NAMES = {'text': 'text', 'image': 'images', 'audio': 'audio'}


class MediaError(ValueError):
    """An image or audio input this request cannot use. The message is safe to return to the caller:
    it names the input by position and never repeats its content."""


# -- client side: anything -> data URL ---------------------------------------------------------------

def _sniff_image(data):
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if data[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'webp'
    return None


def _sniff_audio(data):
    if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
        return 'wav'
    if data[:4] == b'fLaC':
        return 'flac'
    if data[:4] == b'OggS':
        return 'ogg'
    if data[:4] == b'\x1a\x45\xdf\xa3':
        return 'webm'
    if data[:3] == b'ID3' or (len(data) > 1 and data[0] == 0xff and data[1] & 0xe0 == 0xe0):
        return 'mpeg'
    return None


def _read(value, what):
    """Bytes of a local path or an http(s) URL."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if value.startswith(('http://', 'https://')):
        from urllib.request import Request, urlopen
        try:
            with urlopen(Request(value, headers={'User-Agent': 'saina-python'}), timeout=DOWNLOAD_TIMEOUT) as r:
                data = r.read(MAX_MEDIA_BYTES + 1)
        except OSError as exc:
            raise MediaError(f'Cannot download the {what} at {value} ({exc})') from None
    else:
        try:
            with open(os.path.expanduser(value), 'rb') as f:
                data = f.read(MAX_MEDIA_BYTES + 1)
        except OSError as exc:
            raise MediaError(f'Cannot read the {what} {value} ({exc.strerror or exc})') from None
    if len(data) > MAX_MEDIA_BYTES:
        raise MediaError(f'The {what} {value} is larger than {MAX_MEDIA_BYTES:,} bytes')
    return data


def _data_url(media, subtype, data):
    return f'data:{media}/{subtype};base64,' + base64.b64encode(bytes(data)).decode('ascii')


def image_data_url(value):
    """One image as a ``data:image/(png|jpeg|webp);base64,...`` URL.

    ``value`` is a data URL (returned unchanged), a local path, an http(s) URL (downloaded), raw bytes or a
    PIL image (encoded as PNG). Other formats are converted to PNG when Pillow is installed."""
    if isinstance(value, str) and value.startswith('data:'):
        return value
    if hasattr(value, 'save') and hasattr(value, 'mode') and hasattr(value, 'size'):  # a PIL image
        buffer = io.BytesIO()
        value.save(buffer, format='PNG')
        return _data_url('image', 'png', buffer.getvalue())
    if isinstance(value, (str, os.PathLike)):
        data = _read(value, 'image')
    elif isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    else:
        raise MediaError('An image is a data URL, a local path, an http(s) URL, bytes or a PIL image')
    kind = _sniff_image(data)
    if kind:
        return _data_url('image', kind, data)
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as image:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
        return _data_url('image', 'png', buffer.getvalue())
    except ImportError:
        raise MediaError('Send PNG, JPEG or WebP images (install Pillow to convert other formats)') from None
    except Exception:
        raise MediaError('The image could not be read as an image') from None


def _wav(samples, sample_rate):
    """16-bit PCM WAV bytes of a float (-1..1) or int16 array, mono [n] or [n, channels]."""
    import wave
    import numpy as np
    x = np.asarray(samples)
    if x.ndim not in (1, 2) or not x.size:
        raise MediaError('Audio samples must be a non-empty [samples] or [samples, channels] array')
    if x.dtype != np.int16:
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all():
            raise MediaError('Audio samples must be finite')
        x = np.round(np.clip(x, -1.0, 1.0) * 32767).astype('<i2')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as w:
        w.setnchannels(1 if x.ndim == 1 else x.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(x.astype('<i2').tobytes())
    return buffer.getvalue()


def audio_data_url(value):
    """One audio clip as a ``data:audio/<type>;base64,...`` URL (WAV, FLAC, MP3, Ogg or WebM).

    ``value`` is a data URL (returned unchanged), a local path, an http(s) URL (downloaded), raw bytes, or a
    ``(samples, sample_rate)`` pair of a NumPy array and its rate in Hz (encoded as 16-bit WAV)."""
    if isinstance(value, str) and value.startswith('data:'):
        return value
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], (int, float)):
        if not value[1] > 0:
            raise MediaError('The sample rate must be positive')
        return _data_url('audio', 'wav', _wav(*value))
    if isinstance(value, (str, os.PathLike)):
        data = _read(value, 'audio clip')
    elif isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    else:
        raise MediaError('An audio clip is a data URL, a local path, an http(s) URL, bytes or (samples, rate)')
    kind = _sniff_audio(data)
    if not kind:
        raise MediaError('Send WAV, FLAC, MP3, Ogg or WebM audio')
    return _data_url('audio', kind, data)


# -- server side: validation ---------------------------------------------------------------------------

_DATA_URL = re.compile(r'data:(image|audio|video)/([a-z0-9.+-]+)((?:;[a-z0-9_.+-]+=[^;,]*)*);base64', re.I)


def parse_data_url(value, media, allowed, where):
    """``(subtype, bytes)`` of one base64 data URL, checked against the allowed types and the size limit."""
    if not isinstance(value, str) or value[:5].lower() != 'data:':
        raise MediaError(f'{where} must be a base64 data URL (data:{media}/<type>;base64,...)')
    head, comma, text = value.partition(',')
    match = _DATA_URL.fullmatch(head.strip())
    if not comma or not match or match.group(1).lower() != media:
        raise MediaError(f'{where} must be a base64 data URL (data:{media}/<type>;base64,...)')
    subtype = match.group(2).lower()
    if subtype not in allowed:
        raise MediaError(f'{where} has type {media}/{subtype}; accepted types are '
                         + ', '.join(f'{media}/{t}' for t in allowed))
    if len(text) > 4 * math.ceil(MAX_MEDIA_BYTES / 3) + 4:
        raise MediaError(f'{where} is larger than {MAX_MEDIA_BYTES:,} bytes')
    try:
        data = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise MediaError(f'{where} does not hold valid base64') from None
    if len(data) > MAX_MEDIA_BYTES:
        raise MediaError(f'{where} is larger than {MAX_MEDIA_BYTES:,} bytes')
    if not data:
        raise MediaError(f'{where} is empty')
    return subtype, data


def describe_inputs(inputs):
    names = [INPUT_NAMES.get(i, i) for i in inputs]
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


@dataclass(frozen=True)
class RawMedia:
    """Validated, still-encoded media of one request: ``(subtype, bytes)`` per image and per audio clip."""
    images: tuple = ()
    audios: tuple = ()

    def __bool__(self):
        return bool(self.images or self.audios)


def check_media(images=None, audios=None, videos=None, inputs=('text',), model_id='This model'):
    """Validate a request's media fields without decoding pixels or samples; raises MediaError."""
    if videos is not None and (not isinstance(videos, (list, tuple)) or videos):
        raise MediaError('videos are not supported yet')
    for name, items, limit in (('images', images, MAX_IMAGES), ('audios', audios, MAX_AUDIOS)):
        if items is not None and not isinstance(items, (list, tuple)):
            raise MediaError(f'{name} must be a list of base64 data URLs')
        if items and ('image' if name == 'images' else 'audio') not in inputs:
            raise MediaError(f'{model_id} accepts {describe_inputs(inputs)} only')
        if items and len(items) > limit:
            raise MediaError(f'At most {limit} {name} per request')
    return RawMedia(
        tuple(parse_data_url(v, 'image', IMAGE_TYPES, f'images[{i}]') for i, v in enumerate(images or ())),
        tuple(parse_data_url(v, 'audio', AUDIO_TYPES, f'audios[{i}]') for i, v in enumerate(audios or ())))


# -- server side: decoding -----------------------------------------------------------------------------

@dataclass
class Media:
    """What the model reads: RGB images in order (the request's images, then one spectrogram per clip)."""
    images: list = field(default_factory=list)
    sources: list = field(default_factory=list)  # 'image' or 'audio' per entry

    def __bool__(self):
        return bool(self.images)


def decode_image(data, where='image'):
    """An RGB PIL image from PNG, JPEG or WebP bytes; no EXIF rotation and no resize."""
    from PIL import Image
    try:
        probe = Image.open(io.BytesIO(data))
    except Exception:
        raise MediaError(f'{where} could not be read as an image') from None
    with probe:
        if probe.format not in ('PNG', 'JPEG', 'WEBP'):
            raise MediaError(f'{where} is not a PNG, JPEG or WebP image')
        width, height = probe.size
        if width * height > MAX_IMAGE_SOURCE_PIXELS:
            raise MediaError(f'{where} has {width * height:,} pixels; at most {MAX_IMAGE_SOURCE_PIXELS:,} are accepted')
        if max(width, height) > MAX_ASPECT_RATIO * min(width, height):
            raise MediaError(f'{where} is more than {MAX_ASPECT_RATIO} times as wide as it is high, or the reverse')
        try:
            probe.verify()
        except Exception:
            raise MediaError(f'{where} is damaged or truncated') from None
    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.convert('RGB')
    except Exception:
        raise MediaError(f'{where} could not be decoded') from None


def _ffmpeg_decode(data):
    """Float32 WAV bytes of any audio ffmpeg reads (used for WebM and anything libsndfile cannot read)."""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return None
    with tempfile.TemporaryDirectory(prefix='saina-audio-') as tmp:
        source, target = os.path.join(tmp, 'in'), os.path.join(tmp, 'out.wav')
        with open(source, 'wb') as f:
            f.write(data)
        try:
            subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-i', source, '-vn', '-acodec', 'pcm_f32le',
                            '-f', 'wav', target], check=True, timeout=60, capture_output=True)
        except (subprocess.SubprocessError, OSError):
            return None
        with open(target, 'rb') as f:
            return f.read()


def decode_audio(data, where='audio clip'):
    """``(float32 mono samples, sample rate)`` of encoded audio: channels are averaged."""
    import numpy as np
    import soundfile as sf
    try:
        x, rate = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
    except Exception:
        wav = _ffmpeg_decode(data)
        if wav is None:
            raise MediaError(f'{where} could not be decoded') from None
        try:
            x, rate = sf.read(io.BytesIO(wav), dtype='float32', always_2d=True)
        except Exception:
            raise MediaError(f'{where} could not be decoded') from None
    x = x.mean(axis=1)
    if not len(x):
        raise MediaError(f'{where} has no audio samples')
    if not np.isfinite(x).all():
        raise MediaError(f'{where} holds invalid samples')
    return x, int(rate)


def resample(x, rate, target=SAMPLE_RATE):
    """Polyphase resampling to ``target`` Hz (the same method the training corpus used)."""
    import numpy as np
    if rate == target:
        return np.asarray(x, dtype=np.float32)
    from scipy.signal import resample_poly
    g = math.gcd(int(rate), int(target))
    return resample_poly(x, target // g, int(rate) // g).astype(np.float32)


def check_duration(samples, rate, where='audio clip'):
    if len(samples) > MAX_AUDIO_SECONDS * rate:
        raise MediaError(f'{where} is {len(samples) / rate:.1f} s long; at most {MAX_AUDIO_SECONDS:.1f} s '
                         'are accepted (shorten it; nothing is cut off)')


def prepare_waveform(samples, sample_rate):
    """Mono 16 kHz float32 samples of at least MIN_AUDIO_SECONDS, padded with silence at the end."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not len(x):
        raise MediaError('Audio samples must be a non-empty [samples] or [samples, channels] array')
    x = resample(x, sample_rate)
    pad = max(0, int(MIN_AUDIO_SECONDS * SAMPLE_RATE) - len(x))
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
    return x


# -- log-mel spectrogram -------------------------------------------------------------------------------
# Keep identical to the training corpus renderer: power STFT (n_fft 1024, periodic Hann window, hop 160,
# centred with reflect padding), 128 mel bands from 0 to 8 kHz on the Slaney mel scale with area-normalised
# filters (librosa's defaults), 10*log10(power) relative to the clip maximum, clipped to the top 80 dB and
# mapped linearly to 0..255. Low frequencies are the bottom row; one column per hop (100 per second).

def _hz_to_mel(f):
    import numpy as np
    f = np.asarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, f / f_sp)


def _mel_to_hz(m):
    import numpy as np
    m = np.asarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


@lru_cache(maxsize=1)
def mel_filters():
    """[N_MELS, N_FFT // 2 + 1] float32 Slaney-normalised triangular filters."""
    import numpy as np
    fft_freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    mel_pts = _mel_to_hz(np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (mel_pts[2:N_MELS + 2] - mel_pts[:N_MELS]))[:, None]
    return w.astype(np.float32)


@lru_cache(maxsize=1)
def _window():
    import numpy as np
    return np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann


def log_mel(x):
    """[N_MELS, len(x) // HOP] log-mel power in dB relative to the clip maximum, clipped at -TOP_DB."""
    import numpy as np
    n_frames = len(x) // HOP
    xp = np.pad(x, (N_FFT // 2, N_FFT // 2), mode='reflect')
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = xp[idx] * _window()[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = mel_filters() @ spec.T.astype(np.float32)
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db -= db.max()
    return np.maximum(db, -TOP_DB)


def mel_spectrogram(waveform, sample_rate):
    """The spectrogram image of one clip as a uint8 array [128, columns], low frequencies in the last row.

    The clip is averaged to mono, resampled to 16 kHz and padded with silence to at least 1 s, exactly as
    the runtime prepares audio; it is not trimmed (the runtime refuses clips over 10 s instead)."""
    import numpy as np
    x = prepare_waveform(waveform, sample_rate)
    db = log_mel(x)
    return np.round((db + TOP_DB) * (255.0 / TOP_DB)).astype(np.uint8)[::-1]


def mel_spectrogram_png(waveform, sample_rate):
    """8-bit grayscale PNG bytes of :func:`mel_spectrogram`: 128 pixels high, 100 pixels per second."""
    import numpy as np
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(mel_spectrogram(waveform, sample_rate)), mode='L').save(
        buffer, 'PNG', optimize=False, compress_level=6)
    return buffer.getvalue()


def audio_image(data, where='audio clip'):
    """The RGB spectrogram image the model reads for one encoded clip; refuses clips over 10 s."""
    from PIL import Image
    x, rate = decode_audio(data, where)
    check_duration(x, rate, where)
    return Image.fromarray(mel_spectrogram(x, rate), mode='L').convert('RGB')


def decode_media(raw):
    """The images the model reads for validated media: images in request order, then one spectrogram per
    audio clip in request order."""
    media = Media()
    for i, (_, data) in enumerate(raw.images):
        media.images.append(decode_image(data, f'images[{i}]'))
        media.sources.append('image')
    for i, (_, data) in enumerate(raw.audios):
        media.images.append(audio_image(data, f'audios[{i}]'))
        media.sources.append('audio')
    return media


# -- image geometry (no torch) -------------------------------------------------------------------------

def smart_resize(height, width, factor, min_pixels, max_pixels):
    """The size the Qwen vision processor resizes an image to (same rounding as transformers)."""
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise MediaError(f'An image is more than {MAX_ASPECT_RATIO} times as wide as it is high, or the reverse')
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@dataclass(frozen=True)
class ImagePolicy:
    """How a checkpoint's vision processor tiles an image: patch and merge size, and the pixel budget."""
    patch_size: int = 16
    merge_size: int = 2
    min_pixels: int = IMAGE_MIN_PIXELS
    max_pixels: int = IMAGE_MAX_PIXELS

    def grid(self, width, height):
        """``[t, h, w]`` patch grid of one image."""
        h, w = smart_resize(height, width, self.patch_size * self.merge_size, self.min_pixels, self.max_pixels)
        return [1, h // self.patch_size, w // self.patch_size]

    def tokens(self, width, height):
        """Input tokens of one image (one per merged patch, without its start and end markers)."""
        t, h, w = self.grid(width, height)
        return t * h * w // self.merge_size ** 2


def request_media(request, model):
    """Validated media of a parsed request (fields ``images``, ``audios``, ``videos``) for ``model``, still
    encoded; empty when the request has none. Raises MediaError, naming the model when it cannot read them."""
    return check_media(getattr(request, 'images', None), getattr(request, 'audios', None),
                       getattr(request, 'videos', None), getattr(model, 'inputs', ('text',)),
                       getattr(model, 'model_id', 'This model'))
