#!/usr/bin/env python3
"""Generate an original, deterministic text-bearing PNG for the OCR oracle.

The 5x7 glyphs below were drawn for this repository. No fonts, image assets,
network services, or third-party packages are required.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import zlib


TEXT = "MESH 42"
GLYPHS = {
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "4": ("00110", "01010", "10010", "10010", "11111", "00010", "00010"),
    "2": ("11110", "00001", "00001", "01110", "10000", "10000", "11111"),
    " ": ("00000",) * 7,
}
SCALE = 12
MARGIN = 36
CHAR_SPACING = SCALE
WIDTH = 2 * MARGIN + len(TEXT) * (5 * SCALE + CHAR_SPACING) - CHAR_SPACING
HEIGHT = 2 * MARGIN + 7 * SCALE


def chunk(kind: bytes, data: bytes) -> bytes:
    payload = kind + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))


def png_bytes() -> bytes:
    pixels = bytearray(b"\xff" * (WIDTH * HEIGHT * 3))
    for char_index, char in enumerate(TEXT):
        for glyph_y, row in enumerate(GLYPHS[char]):
            for glyph_x, bit in enumerate(row):
                if bit == "0":
                    continue
                x0 = MARGIN + char_index * (5 * SCALE + CHAR_SPACING) + glyph_x * SCALE
                y0 = MARGIN + glyph_y * SCALE
                for y in range(y0, y0 + SCALE):
                    for x in range(x0, x0 + SCALE):
                        offset = 3 * (y * WIDTH + x)
                        pixels[offset : offset + 3] = b"\x00\x00\x00"
    rows = b"".join(
        b"\x00" + pixels[y * WIDTH * 3 : (y + 1) * WIDTH * 3]
        for y in range(HEIGHT)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, level=9))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_bytes(png_bytes())
    print(f"wrote {TEXT!r} OCR oracle fixture: {args.output}")


if __name__ == "__main__":
    main()
