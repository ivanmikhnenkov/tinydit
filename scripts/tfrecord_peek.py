"""Minimal TFRecord + tf.train.Example reader (no TensorFlow). Enough for i1 TFRecords:
records are [u64 len][u32 crc][data][u32 crc]; data is an Example protobuf."""
from __future__ import annotations
import struct


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s; s += 7
        if not c & 0x80: return r, i


def _fields(b):
    """yield (field_no, wire_type, value) for one protobuf message"""
    i, n = 0, len(b)
    while i < n:
        k, i = _varint(b, i); f, w = k >> 3, k & 7
        if w == 0: v, i = _varint(b, i)
        elif w == 1: v, i = b[i:i+8], i + 8
        elif w == 2: l, i = _varint(b, i); v, i = b[i:i+l], i + l
        elif w == 5: v, i = b[i:i+4], i + 4
        else: raise ValueError(f"wire type {w}")
        yield f, w, v


def parse_example(data):
    """-> {name: list_of_values}; bytes_list -> [bytes], float_list -> [float], int64_list -> [int]"""
    out = {}
    for f, w, features in _fields(data):          # Example.features (1)
        if f != 1: continue
        for f2, w2, entry in _fields(features):    # Features.feature map entries (1)
            if f2 != 1: continue
            name, feat = None, None
            for f3, w3, v in _fields(entry):
                if f3 == 1: name = v.decode()
                elif f3 == 2: feat = v
            vals = []
            for f4, w4, lst in _fields(feat):      # Feature.kind: 1 bytes_list, 2 float_list, 3 int64_list
                for f5, w5, v in _fields(lst):
                    if f4 == 1: vals.append(v)
                    elif f4 == 2:
                        if w5 == 2: vals += list(struct.unpack(f"<{len(v)//4}f", v))
                        else: vals.append(struct.unpack("<f", v)[0])
                    elif f4 == 3:
                        if w5 == 2:
                            j = 0
                            while j < len(v):
                                x, j = _varint(v, j); vals.append(x)
                        else: vals.append(v)
            out[name] = vals
    return out


def iter_records(buf):
    """yield example dicts from a (possibly truncated) TFRecord byte buffer"""
    i, n = 0, len(buf)
    while i + 12 <= n:
        (ln,) = struct.unpack("<Q", buf[i:i+8]); i += 12
        if i + ln + 4 > n: return
        yield parse_example(buf[i:i+ln]); i += ln + 4


if __name__ == "__main__":
    import sys, io
    from PIL import Image
    buf = open(sys.argv[1], "rb").read()
    for k, ex in enumerate(iter_records(buf)):
        if k >= 3: break
        print({n: (f"<{len(v)} items, first {len(v[0])}B>" if n == "image" else [x.decode()[:120] if isinstance(x, bytes) else x for x in v][:6]) for n, v in ex.items()})
        im = Image.open(io.BytesIO(ex["image"][0])); print("  image", im.size, im.mode, im.format)
