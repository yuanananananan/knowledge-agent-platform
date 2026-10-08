#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
"$project_dir/.venv/bin/python" "$project_dir/scripts/check_port.py" control-plane
cd "$project_dir/services/go-control-plane"
exec "$project_dir/.venv/bin/python" -m dotenv -f "$project_dir/.env" run -- go run ./cmd/control-plane
