"""文件用途：为雷达实验平台提供严格只读、按用户隔离的 MCP 工具入口。

The server intentionally uses the local stdio transport.  The launching MCP
host supplies ``MCP_USER_ID`` as trusted process configuration; user identity
is never exposed as a model-controlled tool argument.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from mcp.server import MCPServer

from app import session_scope
from app.agent.context import bind_agent_user, reset_agent_user
from app.agent.tools import (
    analyze_job_log,
    diff_train_configs,
    get_experiment_lineage,
    get_train_job_detail,
    list_recent_train_jobs,
)


logger = logging.getLogger(__name__)


def _configured_user_id() -> int:
    """读取由 MCP 宿主可信注入的用户 ID，并拒绝非法配置。"""
    raw = str(os.getenv("MCP_USER_ID", "")).strip()
    try:
        user_id = int(raw)
    except ValueError as exc:
        raise RuntimeError("MCP_USER_ID 必须配置为平台用户的正整数 ID") from exc
    if user_id < 1:
        raise RuntimeError("MCP_USER_ID 必须配置为平台用户的正整数 ID")
    return user_id


def _safe_error(code: str, message: str) -> dict[str, Any]:
    """构造不会泄露服务端内部细节的统一错误响应。"""
    return {"ok": False, "error": {"code": code, "message": message}}


def _invoke_platform_tool(
    tool_instance: Any,
    arguments: dict[str, Any],
    user_id_provider: Callable[[], int],
) -> dict[str, Any]:
    """在服务端身份和独立数据库会话中调用现有平台工具。"""
    try:
        user_id = user_id_provider()
        with session_scope():
            token = bind_agent_user(user_id)
            try:
                raw = tool_instance.invoke(arguments)
            finally:
                reset_agent_user(token)
        result = json.loads(raw)
        if not isinstance(result, dict):
            return _safe_error("invalid_tool_result", "平台工具返回了无效结果")
        return result
    except RuntimeError as exc:
        logger.warning("MCP configuration error: %s", exc)
        return _safe_error("server_misconfigured", str(exc))
    except Exception:
        logger.exception("MCP platform tool failed")
        return _safe_error("tool_failure", "平台工具调用失败，请检查服务端日志")


def create_mcp_server(
    user_id_provider: Callable[[], int] = _configured_user_id,
) -> MCPServer:
    """创建 MCP 服务；可注入用户提供器以便在内存中测试。"""
    server = MCPServer(
        "knowledge-agent-radar-experiments",
        version="0.1.0",
        instructions=(
            "提供多模态雷达实验平台的只读查询和诊断能力。"
            "所有结果均限制在服务端配置的当前用户范围内。"
        ),
    )

    def call(tool_instance: Any, **arguments: Any) -> dict[str, Any]:
        return _invoke_platform_tool(
            tool_instance,
            arguments,
            user_id_provider=user_id_provider,
        )

    @server.tool()
    def list_train_jobs(limit: int = 5, status: str = "") -> dict[str, Any]:
        """查询当前 MCP 用户最近的训练任务，可按状态过滤。"""
        return call(list_recent_train_jobs, limit=limit, status=status)

    @server.tool()
    def inspect_train_job(job_id: str) -> dict[str, Any]:
        """查询当前 MCP 用户某个训练任务的配置、状态和结果。"""
        return call(get_train_job_detail, job_id=job_id)

    @server.tool()
    def diagnose_train_job(job_id: str, max_lines: int = 300) -> dict[str, Any]:
        """分析当前 MCP 用户训练任务的日志，返回错误证据和修复建议。"""
        return call(analyze_job_log, job_id=job_id, max_lines=max_lines)

    @server.tool()
    def compare_train_configs(job_id_1: str, job_id_2: str) -> dict[str, Any]:
        """比较当前 MCP 用户两个训练任务的配置差异。"""
        return call(diff_train_configs, job_id_1=job_id_1, job_id_2=job_id_2)

    @server.tool()
    def inspect_experiment_lineage(job_id: str) -> dict[str, Any]:
        """查询当前 MCP 用户训练任务及其验证任务形成的实验血缘。"""
        return call(get_experiment_lineage, job_id=job_id)

    return server


mcp = create_mcp_server()


def main() -> None:
    """通过标准输入输出把服务作为本地 MCP 子进程运行。"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
