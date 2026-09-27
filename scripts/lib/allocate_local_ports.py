#!/usr/bin/env python3
"""Select distinct OS-assigned loopback ports for an imminent local launch."""
from __future__ import annotations

import argparse
import socket


MAX_PORTS = 16


def allocate_ports(count: int) -> list[int]:
    if type(count) is not int or not 1 <= count <= MAX_PORTS:
        raise ValueError(f"port count must be between 1 and {MAX_PORTS}")
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            sockets.append(listener)
        return [int(listener.getsockname()[1]) for listener in sockets]
    finally:
        for listener in sockets:
            listener.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("count", type=int)
    args = parser.parse_args()
    try:
        ports = allocate_ports(args.count)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(",".join(str(port) for port in ports))


if __name__ == "__main__":
    main()
