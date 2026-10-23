"""Compressed request bodies: `Content-Encoding: gzip` (or `x-gzip`) and `zstd`, one layer, size-bounded.

The official TypeSafe SDKs zstd-compress request bodies of 1 KiB or more by default, so System One clients
need this to send anything but small requests. Decoding stops one byte past the limit, so a small
compressed body cannot expand into a large allocation. Shared by the server and the hosted gateway.
"""
import zlib

try:
    from compression import zstd as _zstd  # Python 3.14+
except ImportError:
    try:
        from backports import zstd as _zstd
    except ImportError:
        _zstd = None


class EncodingError(ValueError):
    """`status` is 415 for an encoding that is not accepted, 413 when too large, 400 when malformed."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def encodings(header):
    """The encodings named by a Content-Encoding header, outermost last; `identity` is no encoding."""
    return [e.strip().lower() for e in (header or '').split(',') if e.strip() and e.strip().lower() != 'identity']


def decode_body(raw, header, limit):
    """The body as sent before compression, at most `limit` bytes. No header: `raw` unchanged."""
    found = encodings(header)
    if not found:
        return raw
    if len(found) > 1:
        raise EncodingError(415, 'Use at most one Content-Encoding')
    encoding = found[0]
    if encoding in ('gzip', 'x-gzip'):
        def decompressor():
            d = zlib.decompressobj(wbits=31)
            return d, lambda data, n: d.decompress(data, n), lambda: d.eof, lambda: d.unused_data
    elif encoding == 'zstd':
        if _zstd is None:
            raise EncodingError(415, 'zstd request bodies need Python 3.14 or the backports-zstd package on this server')

        def decompressor():
            d = _zstd.ZstdDecompressor()
            return d, lambda data, n: d.decompress(data, max_length=n), lambda: d.eof, lambda: d.unused_data
    else:
        raise EncodingError(415, 'Content-Encoding must be gzip or zstd')
    out, data = bytearray(), raw
    try:
        while True:  # gzip members and zstd frames may be concatenated
            _, step, done, rest = decompressor()
            out += step(data, limit + 1 - len(out))
            if len(out) > limit:
                raise EncodingError(413, f'Decompressed request body exceeds {limit} bytes')
            if not done():
                raise EncodingError(400, 'Compressed request body is truncated or too large to decompress')
            data = rest()
            if not data:
                return bytes(out)
    except EncodingError:
        raise
    except Exception:  # zlib.error, zstd.ZstdError: malformed input
        raise EncodingError(400, 'Compressed request body is malformed') from None
