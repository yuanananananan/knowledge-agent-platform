#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
go_path=$(go env GOPATH)

cd "$project_root"
PATH="$go_path/bin:$PATH" protoc \
  -I . \
  --go_out=services/go-control-plane \
  --go_opt=module=knowledge-agent/go-control-plane \
  --go_opt=Mproto/agent/v1/agent_execution.proto=knowledge-agent/go-control-plane/gen/agent/v1 \
  --go-grpc_out=services/go-control-plane \
  --go-grpc_opt=module=knowledge-agent/go-control-plane \
  --go-grpc_opt=Mproto/agent/v1/agent_execution.proto=knowledge-agent/go-control-plane/gen/agent/v1 \
  proto/agent/v1/agent_execution.proto

PATH="$go_path/bin:$PATH" protoc \
  -I . \
  --go_out=services/go-business-api \
  --go_opt=module=knowledge-agent/go-business-api \
  --go_opt=Mproto/agent/v1/agent_execution.proto=knowledge-agent/go-business-api/gen/agent/v1 \
  --go-grpc_out=services/go-business-api \
  --go-grpc_opt=module=knowledge-agent/go-business-api \
  --go-grpc_opt=Mproto/agent/v1/agent_execution.proto=knowledge-agent/go-business-api/gen/agent/v1 \
  proto/agent/v1/agent_execution.proto

.venv/bin/python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  --grpc_python_out=. \
  proto/agent/v1/agent_execution.proto

gofmt -w services/go-control-plane/gen/agent/v1/*.go
gofmt -w services/go-business-api/gen/agent/v1/*.go
