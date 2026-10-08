"""文件用途：实现 Python 到 Go Control Plane 的 HMAC 签名 HTTP 客户端。"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator


class ControlPlaneError(RuntimeError):
    """保留 Control Plane 的 HTTP 状态码和结构化错误响应。"""
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ControlPlaneClient:
    """使用 HMAC 用户身份调用 Go Control Plane 的轻量客户端。"""
    def enabled(self) -> bool:
        return bool(os.getenv("CONTROL_PLANE_URL", "").strip())

    def _base_url(self) -> str:
        value = os.getenv("CONTROL_PLANE_URL", "").strip().rstrip("/")
        if not value:
            raise ControlPlaneError("Go 控制面未配置", 503)
        return value

    def _headers(self, user_id: int) -> dict[str, str]:
        secret = os.getenv("CONTROL_PLANE_AUTH_SECRET", "")
        if len(secret) < 16:
            raise ControlPlaneError("Go 控制面身份密钥未配置", 503)
        stamp = str(int(time.time()))
        raw_user = str(int(user_id))
        signature = hmac.new(
            secret.encode("utf-8"),
            f"{raw_user}:{stamp}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "Accept": "application/json",
            "X-User-ID": raw_user,
            "X-Auth-Timestamp": stamp,
            "X-Auth-Signature": signature,
        }

    def request(
        self,
        user_id: int,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """发送带 HMAC 用户身份的 JSON 请求，并统一转换网络与协议错误。"""
        headers = self._headers(user_id)
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self._base_url() + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read(4 << 20)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise ControlPlaneError(detail or "Go 控制面请求失败", exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ControlPlaneError(f"Go 控制面不可用：{exc}", 503) from exc
        try:
            result = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ControlPlaneError("Go 控制面返回了无效响应", 502) from exc
        if not isinstance(result, dict):
            raise ControlPlaneError("Go 控制面返回格式错误", 502)
        return result

    def create_run(self, user_id: int, message: str, conversation_id: str | None) -> dict[str, Any]:
        """向控制面创建一次新的 Agent 运行。"""
        body: dict[str, Any] = {"message": message}
        if conversation_id:
            body["conversation_id"] = conversation_id
        return self.request(user_id, "POST", "/v1/runs", body)

    def get_run(self, user_id: int, run_id: str) -> dict[str, Any]:
        return self.request(user_id, "GET", f"/v1/runs/{urllib.parse.quote(run_id, safe='')}")

    def list_runs(self, user_id: int, conversation_id: str | None = None, limit: int = 20) -> dict[str, Any]:
        query = {"limit": str(max(1, min(int(limit), 100))), "offset": "0"}
        if conversation_id:
            query["conversation_id"] = conversation_id
        return self.request(user_id, "GET", "/v1/runs?" + urllib.parse.urlencode(query))

    def cancel_run(self, user_id: int, run_id: str) -> dict[str, Any]:
        return self.request(user_id, "DELETE", f"/v1/runs/{urllib.parse.quote(run_id, safe='')}")

    def run_events(self, user_id: int, run_id: str, after: int = 0) -> dict[str, Any]:
        """获取指定序号之后的持久化事件快照，用于断点恢复。"""
        path = (
            f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/events/snapshot"
            f"?after={max(0, int(after))}"
        )
        return self.request(user_id, "GET", path)

    def wait_for_run(self, user_id: int, run_id: str, timeout: float = 180.0) -> dict[str, Any]:
        """轮询运行状态，直到成功、失败、取消或超时。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.get_run(user_id, run_id)
            if run.get("status") in {"succeeded", "failed", "cancelled"}:
                return run
            time.sleep(0.4)
        raise ControlPlaneError("Agent 运行超时", 504)

    def stream_run(
        self,
        user_id: int,
        run_id: str,
        after: int = 0,
        timeout: float = 40.0,
    ) -> Iterator[bytes]:
        """逐行转发控制面的 SSE 事件流，并保证结束时关闭连接。"""
        headers = self._headers(user_id)
        headers["Accept"] = "text/event-stream"
        path = (
            f"/v1/runs/{urllib.parse.quote(run_id, safe='')}/events"
            f"?after={max(0, int(after))}"
        )
        request = urllib.request.Request(self._base_url() + path, headers=headers, method="GET")
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise ControlPlaneError(detail or "Go 事件流请求失败", exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ControlPlaneError(f"Go 控制面不可用：{exc}", 503) from exc
        try:
            while True:
                line = response.readline()
                if not line:
                    break
                yield line
        finally:
            response.close()

    def create_action(self, user_id: int, body: dict[str, Any]) -> dict[str, Any]:
        return self.request(user_id, "POST", "/v1/actions", body)

    def list_actions(self, user_id: int, limit: int = 20) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        return self.request(user_id, "GET", f"/v1/actions?limit={limit}&offset=0")

    def get_action(self, user_id: int, proposal_id: str) -> dict[str, Any]:
        path = f"/v1/actions/{urllib.parse.quote(proposal_id, safe='')}"
        return self.request(user_id, "GET", path)

    def approve_action(self, user_id: int, proposal_id: str) -> dict[str, Any]:
        path = f"/v1/actions/{urllib.parse.quote(proposal_id, safe='')}/approve"
        return self.request(user_id, "POST", path, {})

    def reject_action(self, user_id: int, proposal_id: str) -> dict[str, Any]:
        path = f"/v1/actions/{urllib.parse.quote(proposal_id, safe='')}/reject"
        return self.request(user_id, "POST", path, {})


client = ControlPlaneClient()
