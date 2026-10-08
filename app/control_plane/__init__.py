"""文件用途：导出访问 Go Control Plane 的 Python 客户端辅助能力。

The Go Business API is the browser-facing session gateway. It signs the
authenticated user identity and forwards control-plane operations to Go;
Python remains responsible for LangGraph and domain action execution only.
"""

from .client import ControlPlaneError, client

__all__ = ["ControlPlaneError", "client"]
