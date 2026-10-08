from types import SimpleNamespace

import numpy as np
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.checkpoints import _updated_context
from app.agent.evaluation import versioned_golden_case_ids
from app.agent.service import AgentRunHandle, run_agent
from app.agent.tools import ALL_AGENT_TOOLS
from app.agent.workflow import extract_job_ids, route_request
from app.grpc_server import AgentExecutionServicer
from proto.agent.v1 import agent_execution_pb2


class FakeContext:
    def invocation_metadata(self):
        return (SimpleNamespace(key="x-control-plane-token", value="test-token"),)

    def abort(self, code, details):
        raise RuntimeError(f"{code}: {details}")

    def is_active(self):
        return True


def test_complete_tool_registry_and_real_job_ids():
    names = {tool.name for tool in ALL_AGENT_TOOLS}
    assert len(names) >= 16
    assert {"draft_agent_action", "get_agent_action", "get_train_job_detail"} <= names
    assert extract_job_ids(
        "比较 585db3db398eb9c85ef7896444d1ff64 和 550e8400-e29b-41d4-a716-446655440000"
    ) == [
        "585db3db398eb9c85ef7896444d1ff64",
        "550e8400-e29b-41d4-a716-446655440000",
    ]


def test_structured_context_keeps_baseline_override_and_confirmation():
    state = _updated_context({}, "assistant", "建议以 585db3db398eb9c85ef7896444d1ff64 为基线")
    state = _updated_context(state, "user", "epochs 设置为 20，开始训练")
    assert state["baseline_job_id"] == "585db3db398eb9c85ef7896444d1ff64"
    assert state["config_overrides"]["training.epochs"] == 20
    assert state["confirmations"]["create_train_job"] is True
    assert route_request("开始训练")[0] == "action"
    assert route_request("查询平台可用的骨干网络、分类器和特征算子")[0] == "observe"


def test_grpc_stream_forwards_structured_events(monkeypatch):
    monkeypatch.setenv("CONTROL_PLANE_INTERNAL_TOKEN", "test-token")
    monkeypatch.setattr(
        "app.grpc_server.run_agent_stream",
        lambda **_kwargs: iter([
            {"sequence": 1, "type": "model.delta", "payload": {"content": "答"}},
            {"sequence": 2, "type": "run.completed", "payload": {"answer": "答案"}},
        ]),
    )
    service = AgentExecutionServicer()
    events = list(service.ExecuteRunStream(agent_execution_pb2.ExecuteRunRequest(
        run_id="run_1", conversation_id="conv_1", user_id=1, message="问题"
    ), FakeContext()))
    assert [event.type for event in events] == ["model.delta", "run.completed"]
    assert events[-1].terminal is True


def test_grpc_renders_numpy_image_without_flask_http(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTROL_PLANE_INTERNAL_TOKEN", "test-token")
    source = tmp_path / "sample.npy"
    np.save(source, np.arange(64, dtype=np.float32).reshape(8, 8))

    response = AgentExecutionServicer().RenderImage(
        agent_execution_pb2.RenderImageRequest(path=str(source)), FakeContext()
    )

    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert response.content_type == "image/png"
    assert response.filename == "sample.png"


def test_non_streaming_run_persists_intent_and_completed_tools(monkeypatch):
    result = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "observe_datasets",
                    "args": {"modality": ""},
                    "id": "call-1",
                    "type": "tool_call",
                }],
            ),
            ToolMessage(
                content='{"ok":true}',
                name="observe_datasets",
                tool_call_id="call-1",
            ),
            AIMessage(content="数据集列表"),
        ]
    }
    events = []
    monkeypatch.setattr("app.agent.service.get_or_create_conversation", lambda *_args: None)
    monkeypatch.setattr("app.agent.service.load_messages", lambda *_args: [])
    monkeypatch.setattr("app.agent.service.append_message", lambda *_args: None)
    monkeypatch.setattr(
        "app.agent.service._start_run",
        lambda *_args, **_kwargs: AgentRunHandle("run-test", "conv-test", 1),
    )
    monkeypatch.setattr("app.agent.service._finish_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "app.agent.service._record_event",
        lambda _run, event_type, payload: events.append((event_type, payload)) or {},
    )
    monkeypatch.setattr(
        "app.agent.service.build_agent",
        lambda: SimpleNamespace(invoke=lambda *_args, **_kwargs: result),
    )

    answer, conversation_id = run_agent("查询当前平台的数据集", 1, "conv-test")

    assert answer == "数据集列表"
    assert conversation_id == "conv-test"
    assert any(
        event_type == "workflow.node.completed" and payload["intent"] == "observe"
        for event_type, payload in events
    )
    tool_events = [payload for event_type, payload in events if event_type == "tool.completed"]
    assert tool_events == [{
        "name": "observe_datasets",
        "tool_call_id": "call-1",
        "args": {"modality": ""},
        "output": '{"ok":true}',
    }]


def test_default_versioned_golden_suite_has_40_unique_cases():
    case_ids = versioned_golden_case_ids()
    assert len(case_ids) == 40
    assert len(set(case_ids)) == 40
