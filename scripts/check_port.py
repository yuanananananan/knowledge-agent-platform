#!/usr/bin/env python3
"""Fail fast when a local development port is already occupied."""

import socket
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] in {"grpc", "control-plane"}:
        import os

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        address = (
            os.getenv("AGENT_GRPC_ADDRESS", "127.0.0.1:50051")
            if sys.argv[1] == "grpc"
            else os.getenv("CONTROL_PLANE_ADDR", "127.0.0.1:8081")
        )
        host, raw_port = address.rsplit(":", 1)
        if host in {"", "0.0.0.0", "::"}:
            host = "127.0.0.1"
    elif len(sys.argv) == 3:
        host, raw_port = sys.argv[1:]
    else:
        print("usage: check_port.py HOST PORT | grpc | control-plane", file=sys.stderr)
        return 2
    port = int(raw_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.4)
        if connection.connect_ex((host, port)) == 0:
            print(f"refusing to start: {host}:{port} is already in use", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
