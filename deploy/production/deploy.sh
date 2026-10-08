#!/bin/sh
# 文件用途：校验生产配置并以安全覆盖文件构建、启动全部服务。
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
env_file="$project_root/.env.production"

if [ ! -f "$env_file" ]; then
  echo "缺少 $env_file，请先复制 .env.production.example 并填写真实配置。" >&2
  exit 1
fi

if grep -Eq '替换为|example\.(com|internal)|^LLM_API_KEY=填写' "$env_file"; then
  echo ".env.production 仍包含示例占位符，拒绝上线。" >&2
  exit 1
fi

chmod 600 "$env_file"
cd "$project_root"

docker compose --env-file "$env_file" -f docker-compose.yml -f compose.production.yml config --quiet
docker compose --env-file "$env_file" -f docker-compose.yml -f compose.production.yml pull
docker compose --env-file "$env_file" -f docker-compose.yml -f compose.production.yml build --pull
docker compose --env-file "$env_file" -f docker-compose.yml -f compose.production.yml up -d --remove-orphans
docker compose --env-file "$env_file" -f docker-compose.yml -f compose.production.yml ps
