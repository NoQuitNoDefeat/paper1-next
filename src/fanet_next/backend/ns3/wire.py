"""FANETAI1 canonical little-endian records, with exact integers and binary64 reals.

Ported from paper1-baseline (ns3_adapter/wire.py, commit 8451021) without the
old-dataclass reconstruction helper; this project exchanges plain dict trees.

This is a local transport format, not authentication or an untrusted network API.
No native object layouts, pickle payloads or shared-memory views cross the boundary.
"""

import dataclasses
import math
import struct
import uuid
from enum import IntEnum

MAGIC = b"FANETAI1"
WIRE_VERSION = 2
HEADER = struct.Struct("<8sHHIII16sQQq32s")
MAX_COUNT = 100_000
MAX_DEPTH = 24
MAX_STRING = 1024


class Kind(IntEnum):
    HELLO = 1
    READY = 2
    INIT = 3
    STATE = 4
    PLAN = 5
    RESULT = 6
    STOP = 7
    FINAL = 8
    ACK = 9
    CLOSED = 10
    ERROR = 11


class ProtocolError(RuntimeError):
    """Missing, malformed or incompatible messages invalidate this transport run."""


def plain(value):
    """Convert owned immutable records to the canonical wire tree."""
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if type(value) in (tuple, list):
        return [plain(v) for v in value]
    if type(value) is dict:
        return {k: plain(v) for k, v in value.items()}
    return value


def encode_value(value, *, limit: int = 8 * 1024 * 1024) -> bytes:
    """Encode a bounded canonical tree; *_ns fields are nonnegative signed int64."""
    out = bytearray()

    def put(data):
        if len(out) + len(data) > limit:
            raise ValueError("wire payload exceeds configured byte capacity")
        out.extend(data)

    def item(v, depth=0, name=""):
        if depth > MAX_DEPTH:
            raise ValueError("wire nesting limit")
        if v is None:
            put(b"\x00")
        elif type(v) is bool:
            put(b"\x02" if v else b"\x01")
        elif type(v) is int:
            signed = name.endswith("_ns")
            if not 0 <= v <= (2**63 - 1 if signed else 2**64 - 1):
                raise ValueError(f"wire integer outside range: {name}")
            put(bytes([4 if signed else 3]) + struct.pack("<q" if signed else "<Q", v))
        elif type(v) is float:
            if not math.isfinite(v):
                raise ValueError("wire nonfinite binary64")
            put(b"\x05" + struct.pack("<d", v))
        elif type(v) is str:
            data = v.encode("ascii")
            if len(data) > MAX_STRING or any(c < 32 or c > 126 for c in data):
                raise ValueError("wire strings require bounded printable ASCII")
            put(b"\x06" + struct.pack("<I", len(data)) + data)
        elif type(v) in (list, tuple):
            if len(v) > MAX_COUNT:
                raise ValueError("wire array count exceeds limit")
            put(b"\x07" + struct.pack("<I", len(v)))
            for child in v:
                item(child, depth + 1)
        elif type(v) is dict:
            if len(v) > 256 or any(type(k) is not str for k in v):
                raise ValueError("wire object keys/count")
            put(b"\x08" + struct.pack("<I", len(v)))
            for key in sorted(v):
                item(key, depth + 1)
                item(v[key], depth + 1, key)
        else:
            raise ValueError(f"unsupported wire type: {type(v).__name__}")

    item(plain(value))
    return bytes(out)


def decode_value(data: bytes):
    """Decode into independent host values, rejecting truncation and noncanonical maps."""
    pos = 0

    def take(n):
        nonlocal pos
        if n > len(data) - pos:
            raise ProtocolError("truncated wire value")
        result = data[pos : pos + n]
        pos += n
        return result

    def count(maximum):
        n = struct.unpack("<I", take(4))[0]
        if n > maximum or n > len(data) - pos:
            raise ProtocolError("wire length/count out of bounds")
        return n

    def item(depth=0):
        if depth > MAX_DEPTH:
            raise ProtocolError("wire nesting limit")
        tag = take(1)[0]
        if tag == 0:
            return None
        if tag in (1, 2):
            return tag == 2
        if tag in (3, 4, 5):
            value = struct.unpack({3: "<Q", 4: "<q", 5: "<d"}[tag], take(8))[0]
            if (tag == 4 and value < 0) or (tag == 5 and not math.isfinite(value)):
                raise ProtocolError("invalid wire numeric value")
            return value
        if tag == 6:
            raw = take(count(MAX_STRING))
            if any(c < 32 or c > 126 for c in raw):
                raise ProtocolError("wire string is not printable ASCII")
            return raw.decode("ascii")
        if tag == 7:
            return [item(depth + 1) for _ in range(count(MAX_COUNT))]
        if tag == 8:
            values = {}
            for _ in range(count(256)):
                key = item(depth + 1)
                if type(key) is not str or (values and key <= next(reversed(values))):
                    raise ProtocolError("wire object keys must be unique and sorted")
                values[key] = item(depth + 1)
            return values
        raise ProtocolError("unknown wire value tag")

    value = item()
    if pos != len(data):
        raise ProtocolError("trailing wire value bytes")
    return value


def encode(
    kind, run: str, seq: int, epoch: int, sampled_ns: int, digest: bytes, payload, capacity: int
) -> bytes:
    body = encode_value(payload, limit=capacity - HEADER.size)
    if any(type(v) is not int for v in (seq, epoch, sampled_ns)) or not 0 <= seq < 2**64:
        raise ValueError("invalid message sequence/epoch")
    if not 0 <= epoch < 2**64 or not 0 <= sampled_ns < 2**63:
        raise ValueError("invalid message epoch/time")
    if len(digest) != 32 or uuid.UUID(hex=run).hex != run:
        raise ValueError("invalid run/config identity")
    return (
        HEADER.pack(
            MAGIC,
            WIRE_VERSION,
            int(Kind(kind)),
            0,
            len(body),
            HEADER.size,
            bytes.fromhex(run),
            seq,
            epoch,
            sampled_ns,
            digest,
        )
        + body
    )


def decode(data: bytes, capacity: int) -> tuple:
    if not HEADER.size <= len(data) <= capacity:
        raise ProtocolError("message length outside capacity")
    magic, version, kind, flags, length, size, run, seq, epoch, at, digest = HEADER.unpack(
        data[: HEADER.size]
    )
    if (
        magic != MAGIC
        or version != WIRE_VERSION
        or flags != 0
        or size != HEADER.size
        or length != len(data) - size
        or at < 0
    ):
        raise ProtocolError("incompatible header or payload length")
    try:
        kind = Kind(kind)
    except ValueError as error:
        raise ProtocolError("unknown message kind") from error
    return kind, run.hex(), seq, epoch, at, digest, decode_value(data[size:])
