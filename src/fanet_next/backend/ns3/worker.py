"""Isolated native byte-vector worker; the parent alone owns timeouts and resources.

Ported from paper1-baseline (ns3_adapter/worker.py, commit 8451021)."""

import argparse
import importlib
import struct
import sys
from pathlib import Path

from .wire import encode_value


def read_exact(stream, count):
    data = bytearray()
    while len(data) < count:
        part = stream.read(count - len(data))
        if not part:
            raise EOFError("supervisor pipe closed")
        data.extend(part)
    return bytes(data)


def emit(data):
    sys.stdout.buffer.write(struct.pack("<I", len(data)) + data)
    sys.stdout.buffer.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--shm", type=int, required=True)
    parser.add_argument("--tx", type=int, required=True)
    parser.add_argument("--rx", type=int, required=True)
    args = parser.parse_args()
    directory = args.binding_dir.resolve()
    sys.path.insert(0, str(directory))
    binding = importlib.import_module("fanet_bridge_native")
    if Path(binding.__file__).resolve().parent != directory:
        raise RuntimeError("native binding resolved outside configured build")
    channel = binding.WorkerChannel(args.name, args.run, args.shm, args.tx, args.rx)
    emit(encode_value({"ready": True, "build": list(binding.build_info())}))
    while True:
        length = struct.unpack("<I", read_exact(sys.stdin.buffer, 4))[0]
        if length == 0:
            break
        if length > args.tx:
            raise ValueError("supervisor frame exceeds tx capacity")
        data = read_exact(sys.stdin.buffer, length)
        emit(channel.exchange(data))


if __name__ == "__main__":
    main()
