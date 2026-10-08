#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
"$project_dir/.venv/bin/python" "$project_dir/scripts/check_port.py" grpc
cd "$project_dir"
exec "$project_dir/.venv/bin/python" -m app.grpc_server
