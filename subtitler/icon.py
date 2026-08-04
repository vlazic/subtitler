"""The application icon, drawn and encoded here rather than checked in as a binary.

Non-negotiable 6 rules out Pillow, and the macOS tools that build an `.icns` (`iconutil`,
`sips`) exist only on macOS, which would make the one artifact the maintainer cannot test
also the one he cannot build. Both formats are simple enough to write directly:

* **PNG** is a signature plus length-prefixed, CRC-checked chunks. `zlib` does the only
  hard part.
* **ICNS** is a magic word, a total length, and a flat table of (four-byte type, length,
  payload). Since OS X 10.7 the payload for the `ic07`..`ic14` types may be a PNG file
  verbatim, so the whole encoder is a loop over the sizes above.

The drawing is analytic rather than supersampled: a rounded rectangle has a closed-form
distance function, so one pass over the pixels with the coverage clamped to [0, 1] gives
clean edges at every size for the cost of one square root per pixel. Supersampling a
1024-pixel icon in pure Python would take seconds.

What it depicts: a dark screen with two subtitle lines under it, the second one shorter
than the first, which is what a two-line cue looks like and is the only thing this program
makes.
"""

from __future__ import annotations

import math
import struct
import zlib
from collections.abc import Mapping, Sequence

# The palette, as RGB. Dark enough to read on a light Dock and a light GNOME dash, with one
# warm accent so the icon is not a grey square among grey squares.
BACKDROP = (0x16, 0x1B, 0x22)
SCREEN = (0x25, 0x2E, 0x3A)
CAPTION = (0xF2, 0xF5, 0xF8)
ACCENT = (0xF2, 0xA6, 0x3B)

# The sizes an `.icns` carries, and the type code Apple gives each. `ic11` and `ic12` are
# the retina variants of the 16 and 32 point icons, which is why a 1024-pixel image is not
# needed for a sharp menu bar.
ICNS_TYPES: tuple[tuple[bytes, int], ...] = (
    (b"ic11", 32),
    (b"ic12", 64),
    (b"ic07", 128),
    (b"ic13", 256),
    (b"ic08", 256),
    (b"ic14", 512),
    (b"ic09", 512),
)

# What `install-app` writes into the freedesktop icon theme. 512 is what GNOME's overview
# scales from; 128 is what a small list view picks.
PNG_SIZES: tuple[int, ...] = (128, 256, 512)


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _rounded_coverage(
    x: float, y: float, box: tuple[float, float, float, float], r: float
) -> float:
    """How much of the pixel at (x, y) a rounded rectangle covers, from 0 to 1.

    The signed distance to a rounded rectangle is the distance to the inset rectangle minus
    the corner radius, which is exact everywhere including the corners. Converting it to
    coverage with `0.5 - d` is the standard one-pixel-wide antialias and is why nothing
    here needs to be drawn at 4x and shrunk.
    """
    left, top, right, bottom = box
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half_w, half_h = (right - left) / 2 - r, (bottom - top) / 2 - r
    dx = max(abs(x - cx) - half_w, 0.0)
    dy = max(abs(y - cy) - half_h, 0.0)
    return _clamp(0.5 - (math.hypot(dx, dy) - r))


def _over(dst: list[int], offset: int, colour: Sequence[int], alpha: float) -> None:
    """Source-over compositing of one opaque colour onto an RGBA byte list."""
    if alpha <= 0.0:
        return
    if alpha >= 1.0:
        dst[offset : offset + 4] = [colour[0], colour[1], colour[2], 255]
        return
    base_a = dst[offset + 3] / 255
    out_a = alpha + base_a * (1 - alpha)
    for channel in range(3):
        src = colour[channel] * alpha
        under = dst[offset + channel] * base_a * (1 - alpha)
        dst[offset + channel] = round((src + under) / out_a) if out_a else 0
    dst[offset + 3] = round(out_a * 255)


def pixels(size: int) -> bytes:
    """The icon as `size * size` RGBA bytes."""
    s = float(size)
    buf = [0] * (size * size * 4)

    # (box, radius, colour), painted back to front. Every number is a fraction of the icon
    # so the drawing is resolution-independent and 32 pixels looks like 512 pixels.
    shapes = [
        ((0.04 * s, 0.04 * s, 0.96 * s, 0.96 * s), 0.22 * s, BACKDROP),
        ((0.16 * s, 0.17 * s, 0.84 * s, 0.60 * s), 0.06 * s, SCREEN),
        ((0.16 * s, 0.68 * s, 0.84 * s, 0.755 * s), 0.037 * s, CAPTION),
        ((0.16 * s, 0.80 * s, 0.62 * s, 0.875 * s), 0.037 * s, CAPTION),
        ((0.66 * s, 0.80 * s, 0.84 * s, 0.875 * s), 0.037 * s, ACCENT),
    ]
    for box, radius, colour in shapes:
        left, top, right, bottom = box
        # Only the rows and columns the shape can touch, which keeps a 512-pixel icon to
        # roughly one full pass rather than five.
        for py in range(max(int(top) - 1, 0), min(int(bottom) + 2, size)):
            y = py + 0.5
            base = py * size * 4
            for px in range(max(int(left) - 1, 0), min(int(right) + 2, size)):
                alpha = _rounded_coverage(px + 0.5, y, box, radius)
                if alpha > 0:
                    _over(buf, base + px * 4, colour, alpha)
    return bytes(buf)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def png_bytes(size: int) -> bytes:
    """A truecolour-with-alpha PNG of the icon at `size` pixels."""
    raw = pixels(size)
    stride = size * 4
    # Filter type 0 (None) in front of every scanline. The image is flat colour with soft
    # edges, so a smarter filter would save little and cost a second implementation.
    scanlines = b"".join(b"\x00" + raw[y * stride : (y + 1) * stride] for y in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(scanlines, 9))
        + _chunk(b"IEND", b"")
    )


def icns_bytes(images: Mapping[bytes, bytes] | None = None) -> bytes:
    """The macOS icon file: a header, then one length-prefixed entry per representation."""
    if images is None:
        cache: dict[int, bytes] = {}
        images = {kind: cache.setdefault(size, png_bytes(size)) for kind, size in ICNS_TYPES}
    body = b"".join(
        kind + struct.pack(">I", len(payload) + 8) + payload for kind, payload in images.items()
    )
    return b"icns" + struct.pack(">I", len(body) + 8) + body
