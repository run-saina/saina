"""Image and audio inputs: conversion, validation, limits, decoding and the spectrogram."""
import base64
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from saina import media
from saina.media import MediaError

try:
    import numpy as np
    from PIL import Image
    import scipy  # noqa: F401
    import soundfile  # noqa: F401
    HAVE_MEDIA = True
except ImportError:  # the media extras are not installed
    HAVE_MEDIA = False

PNG = b'\x89PNG\r\n\x1a\n' + b'\0' * 16
JPEG = b'\xff\xd8\xff\xe0' + b'\0' * 16
WEBP = b'RIFF\0\0\0\0WEBPVP8 ' + b'\0' * 8


def data_url(kind, subtype, data):
    return f'data:{kind}/{subtype};base64,' + base64.b64encode(data).decode()


def png(width=64, height=48, color=(200, 30, 30)):
    buffer = io.BytesIO()
    Image.new('RGB', (width, height), color).save(buffer, 'PNG')
    return buffer.getvalue()


def wav(seconds, rate=16000, channels=1, frequency=440.0):
    t = np.arange(int(seconds * rate)) / rate
    x = 0.5 * np.sin(2 * np.pi * frequency * t)
    if channels > 1:
        x = np.stack([x] * channels, axis=1)
    return media._wav(x, rate)


class DataUrlTests(unittest.TestCase):
    def test_bytes_and_paths_become_data_urls_by_content(self):
        for data, subtype in ((PNG, 'png'), (JPEG, 'jpeg'), (WEBP, 'webp')):
            self.assertEqual(media.image_data_url(data), data_url('image', subtype, data))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'photo.bin'
            path.write_bytes(JPEG)
            self.assertEqual(media.image_data_url(str(path)), data_url('image', 'jpeg', JPEG))
            self.assertEqual(media.image_data_url(path), data_url('image', 'jpeg', JPEG))
        url = data_url('image', 'png', PNG)
        self.assertIs(media.image_data_url(url), url)  # data URLs pass through unchanged

    def test_audio_bytes_are_typed_by_content(self):
        for data, subtype in ((b'RIFF\0\0\0\0WAVEfmt ', 'wav'), (b'fLaC\0\0', 'flac'), (b'OggS\0\0', 'ogg'),
                              (b'\x1a\x45\xdf\xa3\0', 'webm'), (b'ID3\x04\0', 'mpeg'), (b'\xff\xfb\x90\0', 'mpeg')):
            self.assertEqual(media.audio_data_url(data), data_url('audio', subtype, data))
        with self.assertRaisesRegex(MediaError, 'WAV, FLAC, MP3, Ogg or WebM'):
            media.audio_data_url(b'not audio at all')

    def test_unreadable_inputs_are_refused_clearly(self):
        with self.assertRaisesRegex(MediaError, 'Cannot read the image'):
            media.image_data_url('/no/such/file.png')
        with self.assertRaisesRegex(MediaError, 'An image is'):
            media.image_data_url(42)
        with patch.object(media, 'MAX_MEDIA_BYTES', 8), tempfile.NamedTemporaryFile() as f:
            f.write(PNG)
            f.flush()
            with self.assertRaisesRegex(MediaError, 'larger than 8 bytes'):
                media.image_data_url(f.name)

    @unittest.skipUnless(HAVE_MEDIA, 'requires saina[media]')
    def test_pil_images_and_arrays_are_encoded(self):
        url = media.image_data_url(Image.new('RGB', (5, 4), (1, 2, 3)))
        self.assertTrue(url.startswith('data:image/png;base64,'))
        with Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))) as image:
            self.assertEqual((image.size, image.getpixel((0, 0))), ((5, 4), (1, 2, 3)))
        gif = io.BytesIO()
        Image.new('RGB', (3, 3)).save(gif, 'GIF')
        self.assertTrue(media.image_data_url(gif.getvalue()).startswith('data:image/png;base64,'))
        url = media.audio_data_url((np.zeros(1600, dtype=np.float32), 16000))
        self.assertTrue(url.startswith('data:audio/wav;base64,'))
        x, rate = media.decode_audio(base64.b64decode(url.split(',', 1)[1]))
        self.assertEqual((len(x), rate), (1600, 16000))


class CheckMediaTests(unittest.TestCase):
    def test_videos_are_not_supported_yet(self):
        for videos in (['data:video/mp4;base64,AAAA'], 'x'):
            with self.assertRaisesRegex(MediaError, '^videos are not supported yet$'):
                media.check_media(videos=videos, inputs=('text', 'image', 'audio'))
        self.assertFalse(media.check_media(videos=[]))  # an empty list is no videos

    def test_models_name_what_they_accept(self):
        image = data_url('image', 'png', PNG)
        clip = data_url('audio', 'wav', b'RIFF')
        with self.assertRaisesRegex(MediaError, '^saina-helm-2-0.8b accepts text only$'):
            media.check_media(images=[image], inputs=('text',), model_id='saina-helm-2-0.8b')
        with self.assertRaisesRegex(MediaError, '^saina-helm-2-0.8b accepts text only$'):
            media.check_media(audios=[clip], inputs=('text',), model_id='saina-helm-2-0.8b')
        with self.assertRaisesRegex(MediaError, '^m accepts text and images only$'):
            media.check_media(images=[image], audios=[clip], inputs=('text', 'image'), model_id='m')
        raw = media.check_media(images=[image], audios=[clip], inputs=('text', 'image', 'audio'))
        self.assertEqual(raw, media.RawMedia((('png', PNG),), (('wav', b'RIFF'),)))
        self.assertFalse(media.check_media(images=[], audios=None, inputs=('text',)))  # nothing sent

    def test_counts_types_sizes_and_encoding(self):
        image = data_url('image', 'png', PNG)
        everything = ('text', 'image', 'audio')
        with self.assertRaisesRegex(MediaError, 'At most 8 images per request'):
            media.check_media(images=[image] * 9, inputs=everything)
        self.assertEqual(len(media.check_media(images=[image] * 8, inputs=everything).images), 8)
        with self.assertRaisesRegex(MediaError, 'At most 4 audios per request'):
            media.check_media(audios=[data_url('audio', 'wav', b'x')] * 5, inputs=everything)
        for value, message in (('https://example.com/a.png', r'images\[0\] must be a base64 data URL'),
                               (data_url('audio', 'wav', b'x'), r'images\[0\] must be a base64 data URL'),
                               (data_url('image', 'gif', b'GIF8'), 'has type image/gif; accepted types are image/png'),
                               ('data:image/png;base64,***', 'does not hold valid base64'),
                               ('data:image/png;base64,', r'images\[0\] is empty'),
                               ('data:image/png,abc', 'must be a base64 data URL')):
            with self.assertRaisesRegex(MediaError, message):
                media.check_media(images=[value], inputs=everything)
        with self.assertRaisesRegex(MediaError, 'images must be a list'):
            media.check_media(images=image, inputs=everything)
        # Browser recordings carry codec parameters; the type is matched without them, case-insensitively.
        raw = media.check_media(audios=['DATA:Audio/WebM;codecs=opus;base64,' + base64.b64encode(b'\x1a').decode()],
                                inputs=everything)
        self.assertEqual(raw.audios, (('webm', b'\x1a'),))
        with patch.object(media, 'MAX_MEDIA_BYTES', 10):
            with self.assertRaisesRegex(MediaError, r'images\[1\] is larger than 10 bytes'):
                media.check_media(images=[data_url('image', 'png', b'x' * 10), data_url('image', 'png', b'x' * 11)],
                                  inputs=everything)

    def test_errors_never_repeat_content(self):
        secret = data_url('image', 'tiff', b'secret patient record')
        with self.assertRaises(MediaError) as caught:
            media.check_media(images=[secret], inputs=('text', 'image'))
        self.assertNotIn(secret.split(',')[1], str(caught.exception))


@unittest.skipUnless(HAVE_MEDIA, 'requires saina[media]')
class DecodeTests(unittest.TestCase):
    def test_images_decode_to_rgb_in_order_then_spectrograms(self):
        raw = media.check_media(images=[data_url('image', 'png', png(64, 48)), data_url('image', 'png', png(30, 20))],
                                audios=[data_url('audio', 'wav', wav(2.0))], inputs=('text', 'image', 'audio'))
        decoded = media.decode_media(raw)
        self.assertEqual(decoded.sources, ['image', 'image', 'audio'])
        self.assertEqual([im.size for im in decoded.images], [(64, 48), (30, 20), (200, 128)])
        self.assertTrue(all(im.mode == 'RGB' for im in decoded.images))

    def test_damaged_or_mislabelled_images_are_refused(self):
        with self.assertRaisesRegex(MediaError, r'images\[0\] could not be read'):
            media.decode_image(PNG, 'images[0]')
        gif = io.BytesIO()
        Image.new('RGB', (3, 3)).save(gif, 'GIF')
        with self.assertRaisesRegex(MediaError, 'not a PNG, JPEG or WebP'):
            media.decode_image(gif.getvalue())
        with patch.object(media, 'MAX_IMAGE_SOURCE_PIXELS', 100):
            with self.assertRaisesRegex(MediaError, 'at most 100 are accepted'):
                media.decode_image(png(20, 20))
        with self.assertRaisesRegex(MediaError, 'more than 200 times as wide'):
            media.decode_image(png(402, 2))

    def test_audio_over_ten_seconds_is_refused_never_cut(self):
        image = media.audio_image(wav(10.0))
        self.assertEqual(image.size, (1000, 128))  # 100 columns per second
        with self.assertRaisesRegex(MediaError, r'audios\[2\] is 10.5 s long; at most 10.0 s'):
            media.audio_image(wav(10.5), 'audios[2]')
        self.assertEqual(media.audio_image(wav(10.0, rate=44100)).size, (1000, 128))  # exactly 10 s at 44.1 kHz

    def test_compressed_long_audio_is_rejected_before_allocating_samples(self):
        buffer = io.BytesIO()
        soundfile.write(buffer, np.zeros(60 * 16000, dtype=np.float32), 16000, format='FLAC')
        compressed = buffer.getvalue()
        self.assertLess(len(compressed), 3000)  # a tiny upload must not expand into the full minute of PCM
        with patch.object(soundfile.SoundFile, 'read') as read, patch.object(media, '_ffmpeg_decode') as fallback:
            with self.assertRaisesRegex(MediaError, r'audios\[2\] is 60.0 s long; at most 10.0 s'):
                media.audio_image(compressed, 'audios[2]')
        read.assert_not_called()
        fallback.assert_not_called()

    def test_audio_reads_are_bounded_even_with_inaccurate_duration_metadata(self):
        with patch.object(soundfile, 'SoundFile') as open_audio, patch.object(media, '_ffmpeg_decode') as fallback:
            source = open_audio.return_value.__enter__.return_value
            source.samplerate, source.channels, source.frames = 16000, 2, 160000
            source.read.return_value = np.zeros((160001, 2), dtype=np.float32)
            with self.assertRaisesRegex(MediaError, r'audios\[1\].*at most 10.0 s'):
                media.decode_audio(b'encoded audio', 'audios[1]')
            source.read.assert_called_once_with(frames=160001, dtype='float32', always_2d=True)
        fallback.assert_not_called()

    def test_decoded_audio_size_is_checked_before_allocating_samples(self):
        with patch.object(media, 'MAX_DECODED_AUDIO_BYTES', 1000), \
                patch.object(soundfile.SoundFile, 'read') as read, patch.object(media, '_ffmpeg_decode') as fallback:
            with self.assertRaisesRegex(MediaError, r'audios\[0\].*decoded audio size limit'):
                media.decode_audio(wav(1.0), 'audios[0]')
        read.assert_not_called()
        fallback.assert_not_called()

    def test_native_audio_keeps_samples_and_rejects_one_excess_frame(self):
        data = wav(1.5, rate=44100, channels=2)
        expected, rate = soundfile.read(io.BytesIO(data), dtype='float32', always_2d=True)
        actual, actual_rate = media.decode_audio(data)
        self.assertEqual(actual_rate, rate)
        np.testing.assert_array_equal(actual, expected.mean(axis=1))
        with self.assertRaisesRegex(MediaError, 'at most 10.0 s'):
            media.decode_audio(media._wav(np.zeros(441001), 44100))

    def test_audio_is_mono_16k_and_short_clips_are_padded(self):
        self.assertEqual(media.audio_image(wav(0.25)).size, (100, 128))  # padded to 1 s
        stereo, mono = media.audio_image(wav(1.5, channels=2)), media.audio_image(wav(1.5))
        self.assertEqual(np.asarray(stereo).tolist(), np.asarray(mono).tolist())
        self.assertEqual(media.audio_image(wav(1.5, rate=44100)).size, (150, 128))
        with self.assertRaisesRegex(MediaError, 'could not be decoded'):
            media.audio_image(b'RIFF\0\0\0\0WAVEjunk')

    def test_webm_and_mp3_decode_when_ffmpeg_is_available(self):
        import shutil
        import subprocess
        if not shutil.which('ffmpeg'):
            self.skipTest('ffmpeg not installed')
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'in.wav').write_bytes(wav(1.2, rate=48000))
            for name, args in (('out.webm', ['-c:a', 'libopus']), ('out.mp3', [])):
                subprocess.run(['ffmpeg', '-v', 'error', '-i', str(Path(d) / 'in.wav'), *args, str(Path(d) / name)],
                               check=True)
                url = media.audio_data_url(str(Path(d) / name))
                raw = media.check_media(audios=[url], inputs=('text', 'image', 'audio'))
                image = media.decode_media(raw).images[0]
                self.assertEqual(image.size[1], 128)
                self.assertAlmostEqual(image.size[0], 120, delta=8)  # codec padding may add a few frames

    def test_ffmpeg_stops_long_audio_and_preserves_valid_audio(self):
        import shutil
        import subprocess
        if not shutil.which('ffmpeg'):
            self.skipTest('ffmpeg not installed')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for seconds in (10, 60):
                # FLAC supplies a compact long input while forcing the fallback below exercises the
                # same FFmpeg command used for WebM. Native rate/channels must remain unchanged.
                source = root / f'{seconds}.flac'
                soundfile.write(source, np.zeros((seconds * 16000, 2), dtype=np.float32), 16000)
                encoded = source.read_bytes()
                decoded = media._ffmpeg_decode(encoded)
                info = soundfile.info(io.BytesIO(decoded))
                self.assertEqual((info.samplerate, info.channels), (16000, 2))
                self.assertEqual(info.frames, min(seconds, 11) * 16000)
                self.assertLess(len(decoded), 1_500_000)
                real_read = media._read_audio

                def fallback_only(data, where, **kwargs):
                    if data == encoded:
                        raise RuntimeError('force FFmpeg fallback')
                    return real_read(data, where, **kwargs)

                with patch.object(media, '_read_audio', side_effect=fallback_only):
                    if seconds == 10:
                        samples, rate = media.decode_audio(encoded)
                        self.assertEqual((len(samples), rate), (160000, 16000))
                    else:
                        with self.assertRaisesRegex(MediaError, r'audios\[3\] is longer than 10.0 s'):
                            media.decode_audio(encoded, 'audios[3]')
            # Valid compressed input must yield the same native samples as the previous decoder.
            (root / 'in.wav').write_bytes(wav(1.2, rate=48000, channels=2))
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(root / 'in.wav'), '-c:a', 'libopus',
                            str(root / 'clip.webm')], check=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(root / 'clip.webm'), '-vn',
                            '-acodec', 'pcm_f32le', str(root / 'reference.wav')], check=True)
            actual, rate = media.decode_audio((root / 'clip.webm').read_bytes())
            expected, expected_rate = soundfile.read(root / 'reference.wav', dtype='float32', always_2d=True)
            self.assertEqual(rate, expected_rate)
            np.testing.assert_array_equal(actual, expected.mean(axis=1))

    def test_ffmpeg_output_size_limit_never_accepts_a_truncated_clip(self):
        import shutil
        if not shutil.which('ffmpeg'):
            self.skipTest('ffmpeg not installed')
        with patch.object(media, 'MAX_DECODED_AUDIO_BYTES', 4096):
            with self.assertRaisesRegex(MediaError, r'audios\[0\].*decoded audio size limit'):
                media._ffmpeg_decode(wav(1.0), 'audios[0]')


@unittest.skipUnless(HAVE_MEDIA, 'requires saina[media]')
class MelSpectrogramTests(unittest.TestCase):
    def reference(self, x):
        """An independent float64 rendering of the documented definition."""
        x = np.asarray(x, dtype=np.float64)
        n = len(x) // 160
        xp = np.pad(x, 512, mode='reflect')
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1024) / 1024)
        frames = np.stack([xp[t * 160:t * 160 + 1024] * window for t in range(n)])
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        mel = media.mel_filters().astype(np.float64) @ power.T
        db = 10 * np.log10(np.maximum(mel, 1e-10))
        db = np.maximum(db - db.max(), -80)
        return np.round((db + 80) * 255 / 80)[::-1]

    def test_definition_shape_orientation_and_range(self):
        t = np.arange(32000) / 16000
        low, high = np.sin(2 * np.pi * 200 * t), np.sin(2 * np.pi * 4000 * t)
        for x in (low, high, 0.3 * low + 0.01 * np.random.default_rng(0).standard_normal(len(t))):
            image = media.mel_spectrogram(x, 16000)
            self.assertEqual((image.shape, image.dtype), ((128, 200), np.uint8))
            self.assertEqual(image.max(), 255)
            # float32 summation order may move a level by one, never more.
            self.assertLessEqual(np.abs(image.astype(int) - self.reference(x.astype(np.float32))).max(), 1)
        # Low frequencies are the bottom rows.
        self.assertGreater(np.argmax(media.mel_spectrogram(low, 16000)[:, 100]), 100)
        self.assertLess(np.argmax(media.mel_spectrogram(high, 16000)[:, 100]), 40)

    def test_filters_are_slaney_scale_and_area_normalised(self):
        filters = media.mel_filters()
        self.assertEqual(filters.shape, (128, 513))
        hz = media._mel_to_hz(np.linspace(media._hz_to_mel(0), media._hz_to_mel(8000), 130))
        self.assertAlmostEqual(float(hz[-1]), 8000, places=3)
        self.assertAlmostEqual(float(media._hz_to_mel(1000)), 15.0)  # Slaney: linear to 1 kHz at 200/3 Hz per mel
        try:
            import librosa
        except ImportError:
            return
        expected = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=0, fmax=8000, htk=False, norm='slaney')
        self.assertTrue(np.allclose(filters, expected, atol=1e-6))

    def test_png_is_8_bit_grayscale_100_px_per_second(self):
        x = np.sin(2 * np.pi * 440 * np.arange(24000) / 16000)
        data = media.mel_spectrogram_png(x, 16000)
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual((image.format, image.mode, image.size), ('PNG', 'L', (150, 128)))
            self.assertEqual(np.asarray(image).tolist(), media.mel_spectrogram(x, 16000).tolist())
        # Resampled to 16 kHz first: the same tone at 48 kHz gives the same geometry.
        x48 = np.sin(2 * np.pi * 440 * np.arange(72000) / 48000)
        self.assertEqual(media.mel_spectrogram(x48, 48000).shape, (128, 150))


class GeometryTests(unittest.TestCase):
    def test_tokens_follow_the_vision_processor(self):
        policy = media.ImagePolicy()
        self.assertEqual(policy.grid(900, 700), [1, 44, 56])
        self.assertEqual(policy.tokens(900, 700), 44 * 56 // 4)
        self.assertEqual(policy.grid(3000, 2000), [1, 64, 96])  # read at up to 1.6 megapixels
        self.assertEqual(policy.tokens(1000, 128), 8 * 62 // 4)  # a 10 s spectrogram: 124 tokens
        self.assertEqual(policy.grid(100, 128), [1, 20, 16])  # a 1 s spectrogram is scaled up to 65,536 pixels
        self.assertEqual(policy.tokens(100, 128), 80)
        try:
            from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize
        except ImportError:
            return
        import random
        rng = random.Random(0)
        for _ in range(2000):
            h, w = rng.randint(1, 6000), rng.randint(1, 6000)
            if max(h, w) / min(h, w) > 200:
                continue
            self.assertEqual(media.smart_resize(h, w, 32, 65536, 1638400), smart_resize(h, w, 32, 65536, 1638400))

    def test_images_go_before_the_text_of_the_user_turn(self):
        from saina.prompt import insert_images
        from saina.errors import InputTooLong

        class Tok:
            vocab = {'<|vision_start|>': 1, '<|image_pad|>': 2, '<|vision_end|>': 3}
            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text]
            def convert_tokens_to_ids(self, token):
                return self.vocab[token]
        ids = Tok().encode('<|im_start|>user\nhello<|im_end|>\nDecision q0:')
        out = insert_images(Tok(), ids, [2, 1], 1000)
        start = len('<|im_start|>user\n')
        self.assertEqual(out[start:start + 7], [1, 2, 2, 3, 1, 2, 3])
        self.assertEqual(out[:start] + out[start + 7:], ids)
        self.assertEqual(insert_images(Tok(), ids, [], 0), ids)
        with self.assertRaises(InputTooLong):
            insert_images(Tok(), ids, [2, 1], len(ids) + 6)


if __name__ == '__main__':
    unittest.main()
