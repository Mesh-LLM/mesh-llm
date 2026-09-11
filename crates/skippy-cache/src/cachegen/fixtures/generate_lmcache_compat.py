# SPDX-License-Identifier: Apache-2.0
"""Generate language-neutral compatibility fixtures for the Rust CacheGen port.

This is a scalar transcription of LMCache revision
b5d109ea99a89b4d8a670ee4fc2e8cb76411ee5c. The relevant upstream files are:

* lmcache/storage_backend/serde/cachegen_encoder.py
* lmcache/storage_backend/serde/cachegen_decoder.py
* csrc/cuda/ac_enc.cu
* csrc/cuda/ac_dec.cu
* csrc/cuda/cal_cdf.cu

Run from this directory with Python 3.11 or newer. It needs no third-party
packages and deliberately shares no Rust implementation code.
"""

from pathlib import Path
import math
import struct

MAGIC = b"LCG1"
FIXTURE_MAGIC = b"LCFX"
MAX_BINS = 32
CDF_LEN = MAX_BINS + 1
CDF_TOTAL = 1 << 16
QUARTER = 0x40000000
HALF = 0x80000000
THREE_QUARTERS = 0xC0000000
MASK32 = 0xFFFFFFFF


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f16_bytes(value: float) -> bytes:
    return struct.pack("<e", value)


def f16_value(raw: bytes) -> float:
    return struct.unpack("<e", raw)[0]


class BitWriter:
    def __init__(self) -> None:
        self.register = 0
        self.bits = 0
        self.output = bytearray()

    def repeated(self, bit: int, count: int) -> None:
        while count:
            take = min(count, 32 - self.bits)
            self.register = (self.register << take) & MASK32
            if bit:
                self.register |= MASK32 if take == 32 else (1 << take) - 1
            self.bits += take
            count -= take
            if self.bits == 32:
                self.output.extend(struct.pack(">I", self.register))
                self.register = 0
                self.bits = 0

    def append(self, bit: int, pending: int) -> None:
        self.repeated(bit, 1)
        self.repeated(1 - bit, pending)

    def finish(self) -> bytes:
        if self.bits:
            self.register = (self.register << (32 - self.bits)) & MASK32
            self.output.extend(struct.pack(">I", self.register)[: (self.bits + 7) // 8])
        return bytes(self.output)


class BitReader:
    def __init__(self, stream: bytes) -> None:
        self.stream = stream
        self.bit = 0

    def read_bit(self) -> int:
        byte = self.stream[self.bit // 8] if self.bit // 8 < len(self.stream) else 0
        value = (byte >> (7 - self.bit % 8)) & 1
        self.bit += 1
        return value

    def read_u32(self) -> int:
        value = 0
        for _ in range(32):
            value = ((value << 1) | self.read_bit()) & MASK32
        return value


def encode_channel(symbols, rows, channels, channel, cdf) -> bytes:
    low = 0
    high = MASK32
    pending = 0
    writer = BitWriter()
    for row in range(rows):
        symbol = symbols[row * channels + channel]
        span = high - low + 1
        c_low = cdf[symbol]
        c_high = CDF_TOTAL if symbol == MAX_BINS - 1 else cdf[symbol + 1]
        high = ((low - 1) + ((span * c_high) >> 16)) & MASK32
        low = (low + ((span * c_low) >> 16)) & MASK32
        while True:
            if high < HALF:
                writer.append(0, pending)
                pending = 0
            elif low >= HALF:
                writer.append(1, pending)
                pending = 0
            elif low >= QUARTER and high < THREE_QUARTERS:
                pending += 1
                low = (low << 1) & 0x7FFFFFFF
                high = ((high << 1) | 0x80000001) & MASK32
                continue
            else:
                break
            low = (low << 1) & MASK32
            high = ((high << 1) | 1) & MASK32
    pending += 1
    writer.append(1 if low >= QUARTER else 0, pending)
    return writer.finish()


def decode_channel(stream, rows, channels, channel, cdf, output) -> None:
    reader = BitReader(stream)
    low = 0
    high = MASK32
    value = reader.read_u32()
    for row in range(rows):
        span = high - low + 1
        count = (((value - low + 1) * CDF_TOTAL) - 1) // span
        left = 0
        right = MAX_BINS
        while left + 1 < right:
            middle = (left + right) // 2
            if cdf[middle] < count:
                left = middle
            elif cdf[middle] > count:
                right = middle
            else:
                left = middle
                break
        symbol = left
        output[row * channels + channel] = symbol
        if row + 1 == rows:
            break
        c_low = cdf[symbol]
        c_high = CDF_TOTAL if symbol == MAX_BINS - 1 else cdf[symbol + 1]
        high = ((low - 1) + ((span * c_high) >> 16)) & MASK32
        low = (low + ((span * c_low) >> 16)) & MASK32
        while True:
            if low >= HALF or high < HALF:
                low = (low << 1) & MASK32
                high = ((high << 1) | 1) & MASK32
                value = ((value << 1) | reader.read_bit()) & MASK32
            elif low >= QUARTER and high < THREE_QUARTERS:
                low = ((low << 1) & 0x7FFFFFFF) & MASK32
                high = ((high << 1) | 0x80000001) & MASK32
                value = (value - QUARTER) & MASK32
                value = ((value << 1) | reader.read_bit()) & MASK32
            else:
                break


def quantize(values, rows, channels, bins):
    center = f32(bins // 2 - 1)
    symbols = []
    maxes = []
    for row in range(rows):
        values_row = values[row * channels : (row + 1) * channels]
        maximum = max(abs(value) for value in values_row)
        maxes.append(maximum)
        if maximum == 0.0:
            symbols.extend([int(center)] * channels)
            continue
        factor = f32(center / maximum)
        for value in values_row:
            scaled = f32(value * factor)
            shifted = f32(scaled + center)
            symbols.append(max(0, min(int(center * 2), round(shifted))))
    return symbols, maxes


def calculate_cdf(symbols, rows, channels):
    cdfs = []
    for channel in range(channels):
        histogram = [0] * CDF_LEN
        for row in range(rows):
            histogram[symbols[row * channels + channel] + 1] += 1
        running = 0
        for index in range(1, CDF_LEN):
            count = histogram[index]
            histogram[index] += running
            running += count
        cdf = [((0xFFFF - MAX_BINS) * count // running) + index for index, count in enumerate(histogram)]
        assert cdf[0] == 0 and cdf[-1] == 0xFFFF
        assert all(left < right for left, right in zip(cdf, cdf[1:]))
        cdfs.append(cdf)
    return cdfs


def encode(raw: bytes, rows: int, channels: int, bins: int) -> bytes:
    values = [f32(f16_value(raw[index : index + 2])) for index in range(0, len(raw), 2)]
    symbols, maxes = quantize(values, rows, channels, bins)
    cdfs = calculate_cdf(symbols, rows, channels)
    streams = [encode_channel(symbols, rows, channels, channel, cdfs[channel]) for channel in range(channels)]
    payload = bytearray(MAGIC)
    payload.extend(bytes((bins, 0)))
    payload.extend(struct.pack("<HII", rows, channels, sum(map(len, streams))))
    payload.extend(b"".join(struct.pack("<f", maximum) for maximum in maxes))
    payload.extend(b"".join(struct.pack("<H", value) for cdf in cdfs for value in cdf))
    payload.extend(b"".join(struct.pack("<H", len(stream)) for stream in streams))
    payload.extend(b"".join(streams))
    return bytes(payload)


def decode(payload: bytes) -> bytes:
    assert payload[:4] == MAGIC and payload[5] == 0
    bins = payload[4]
    rows, channels, stream_len = struct.unpack("<HII", payload[6:16])
    cursor = 16
    maxes = [struct.unpack("<f", payload[cursor + index * 4 : cursor + index * 4 + 4])[0] for index in range(rows)]
    cursor += rows * 4
    cdfs = []
    for _ in range(channels):
        cdfs.append(list(struct.unpack("<" + "H" * CDF_LEN, payload[cursor : cursor + CDF_LEN * 2])))
        cursor += CDF_LEN * 2
    lengths = list(struct.unpack("<" + "H" * channels, payload[cursor : cursor + channels * 2]))
    cursor += channels * 2
    assert sum(lengths) == stream_len and cursor + stream_len == len(payload)
    symbols = [0] * (rows * channels)
    stream_cursor = cursor
    for channel, length in enumerate(lengths):
        stream = payload[stream_cursor : stream_cursor + length]
        decode_channel(stream, rows, channels, channel, cdfs[channel], symbols)
        stream_cursor += length
    center = f32(bins // 2 - 1)
    raw = bytearray()
    for row, maximum in enumerate(maxes):
        for channel in range(channels):
            centered = f32(f32(symbols[row * channels + channel]) - center)
            normalized = f32(centered / center)
            value = f32(normalized * maximum)
            raw.extend(f16_bytes(value))
    return bytes(raw)


def input_bytes(rows: int, channels: int) -> bytes:
    raw = bytearray()
    for row in range(rows):
        for channel in range(channels):
            phase = f32(f32(row * channels + channel) * f32(0.03125))
            value = f32(f32(math.sin(phase)) * f32(0.75))
            value = f32(value + f32(f32(channel - 3) * f32(0.015625)))
            raw.extend(f16_bytes(value))
    return bytes(raw)


def write_fixture(path: Path, bins: int) -> None:
    rows = 17
    channels = 8
    raw = input_bytes(rows, channels)
    encoded = encode(raw, rows, channels, bins)
    decoded = decode(encoded)
    fixture = FIXTURE_MAGIC + struct.pack("<III", len(raw), len(encoded), len(decoded)) + raw + encoded + decoded
    path.write_bytes(fixture)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    write_fixture(root / "lmcache_b5d109e_bins16.bin", 16)
    write_fixture(root / "lmcache_b5d109e_bins32.bin", 32)
