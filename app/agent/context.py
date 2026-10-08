"""文件用途：通过 ContextVar 在一次 Agent 调用中安全传递当前用户和会话身份。"""

from contextvars import ContextVar, Token


_current_user_id: ContextVar[int | None] = ContextVar(
    "agent_current_user_id",
    default=None,
)
_current_conversation_id: ContextVar[str | None] = ContextVar(
    "agent_current_conversation_id", default=None
)


def bind_agent_user(user_id: int, conversation_id: str | None = None) -> tuple[Token, Token]:
    """Bind the authenticated user to the current Agent execution context."""
    return _current_user_id.set(int(user_id)), _current_conversation_id.set(conversation_id)


# 使用 bind 返回的 Token 恢复调用前的用户和会话上下文。
def reset_agent_user(token: tuple[Token, Token] | Token) -> None:
    if isinstance(token, tuple):
        user_token, conversation_token = token
        _current_user_id.reset(user_token)
        _current_conversation_id.reset(conversation_token)
    else:
        _current_user_id.reset(token)


# 返回当前 Agent 调用绑定的用户 ID，未绑定时拒绝继续执行。
def get_agent_user_id() -> int:
    user_id = _current_user_id.get()
    if user_id is None:
        raise RuntimeError("Agent 工具缺少已认证的用户上下文")
    return user_id


# 返回当前 Agent 调用绑定的可选会话 ID。
def get_agent_conversation_id() -> str | None:
    return _current_conversation_id.get()
