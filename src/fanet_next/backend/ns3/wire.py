"""FANETAI1 canonical little-endian records, with exact integers and binary64 reals.

Ported from paper1-baseline (ns3_adapter/wire.py, commit 8451021) without the
old-dataclass reconstruction helper; this project exchanges plain dict trees.

This is a local transport format, not authentication or an untrusted network API.
No native object layouts, pickle payloads or shared-memory views cross the boundary.
"""

import dataclasses
import math
import re
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


_NONPRINTABLE = re.compile(rb"[^\x20-\x7e]")
_U32, _U64, _I64, _F64 = (struct.Struct(f).pack for f in ("<I", "<Q", "<q", "<d"))
_U32_AT, _U64_AT, _I64_AT, _F64_AT = (struct.Struct(f).unpack_from for f in ("<I", "<Q", "<q", "<d"))
_INT_MAX = {True: 2**63 - 1, False: 2**64 - 1}


def encode_value(value, *, limit: int = 8 * 1024 * 1024) -> bytes:
    """Encode a bounded canonical tree; *_ns fields are nonnegative signed int64."""
    out = bytearray()
    put = out.extend

    def item(v, depth, name):
        if depth > MAX_DEPTH:
            raise ValueError("wire nesting limit")
        t = type(v)
        if t is float:
            if not math.isfinite(v):
                raise ValueError("wire nonfinite binary64")
            put(b"\x05")
            put(_F64(v))
        elif t is int:
            signed = name.endswith("_ns")
            if not 0 <= v <= _INT_MAX[signed]:
                raise ValueError(f"wire integer outside range: {name}")
            put(b"\x04" if signed else b"\x03")
            put(_I64(v) if signed else _U64(v))
        elif t is dict:
            if len(v) > 256 or any(type(k) is not str for k in v):
                raise ValueError("wire object keys/count")
            put(b"\x08")
            put(_U32(len(v)))
            for key in sorted(v):
                item(key, depth + 1, "")
                item(v[key], depth + 1, key)
        elif t is list or t is tuple:
            if len(v) > MAX_COUNT:
                raise ValueError("wire array count exceeds limit")
            put(b"\x07")
            put(_U32(len(v)))
            for child in v:
                item(child, depth + 1, "")
        elif t is str:
            data = v.encode("ascii")
            if len(data) > MAX_STRING or _NONPRINTABLE.search(data):
                raise ValueError("wire strings require bounded printable ASCII")
            put(b"\x06")
            put(_U32(len(data)))
            put(data)
        elif v is None:
            put(b"\x00")
        elif t is bool:
            put(b"\x02" if v else b"\x01")
        elif dataclasses.is_dataclass(v):
            item(plain(v), depth, name)
        else:
            raise ValueError(f"unsupported wire type: {t.__name__}")
        if len(out) > limit:
            raise ValueError("wire payload exceeds configured byte capacity")

    item(value, 0, "")
    return bytes(out)


def decode_value(data: bytes):
    """Decode into independent host values, rejecting truncation and noncanonical maps."""
    data = bytes(data)
    size = len(data)
    pos = 0
    strings: dict[bytes, str] = {}  # map keys repeat; decode each distinct string once

    def need(n):
        if n > size - pos:
            raise ProtocolError("truncated wire value")

    def count(maximum):
        nonlocal pos
        need(4)
        n = _U32_AT(data, pos)[0]
        pos += 4
        if n > maximum or n > size - pos:
            raise ProtocolError("wire length/count out of bounds")
        return n

    def item(depth):
        nonlocal pos
        if depth > MAX_DEPTH:
            raise ProtocolError("wire nesting limit")
        need(1)
        tag = data[pos]
        pos += 1
        if tag == 5:
            need(8)
            value = _F64_AT(data, pos)[0]
            pos += 8
            if not math.isfinite(value):
                raise ProtocolError("invalid wire numeric value")
            return value
        if tag == 3 or tag == 4:
            need(8)
            value = (_U64_AT if tag == 3 else _I64_AT)(data, pos)[0]
            pos += 8
            if value < 0:
                raise ProtocolError("invalid wire numeric value")
            return value
        if tag == 8:
            values = {}
            last = None
            for _ in range(count(256)):
                key = item(depth + 1)
                if type(key) is not str or (last is not None and key <= last):
                    raise ProtocolError("wire object keys must be unique and sorted")
                values[key] = item(depth + 1)
                last = key
            return values
        if tag == 7:
            return [item(depth + 1) for _ in range(count(MAX_COUNT))]
        if tag == 6:
            n = count(MAX_STRING)
            raw = data[pos:pos + n]
            pos += n
            text = strings.get(raw)
            if text is None:
                if _NONPRINTABLE.search(raw):
                    raise ProtocolError("wire string is not printable ASCII")
                text = strings[raw] = raw.decode("ascii")
            return text
        if tag == 0:
            return None
        if tag == 1 or tag == 2:
            return tag == 2
        raise ProtocolError("unknown wire value tag")

    value = item(0)
    if pos != size:
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
