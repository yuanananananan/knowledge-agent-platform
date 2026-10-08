"""文件用途：实现供 Go 服务调用的私有 Python Agent gRPC Server。"""

from __future__ import annotations

import hmac
import io
import json
import logging
import os
from concurrent import futures

import grpc

from app import init_database, session_scope
from app.agent.actions import (
    ActionConflictError,
    ActionValidationError,
    approve_and_execute_action,
    create_action_proposal,
)
from app.agent.service import AgentRunConflictError, run_agent, run_agent_stream
from proto.agent.v1 import agent_execution_pb2
from proto.agent.v1 import agent_execution_pb2_grpc


LOGGER = logging.getLogger(__name__)
TOKEN_METADATA_KEY = "x-control-plane-token"


# 从 gRPC 调用元数据中按不区分大小写的键读取值。
def _metadata_value(context: grpc.ServicerContext, key: str) -> str:
    for item in context.invocation_metadata():
        if item.key.lower() == key:
            return item.value
    return ""


class AgentExecutionServicer(agent_execution_pb2_grpc.AgentExecutionServiceServicer):
    """实现 Agent 执行、动作、计算、知识库、图片和健康检查 RPC。"""
    def _authenticate(self, context: grpc.ServicerContext) -> None:
        """使用共享令牌认证来自 Go 服务的私有调用。"""
        expected = os.getenv("CONTROL_PLANE_INTERNAL_TOKEN", "")
        supplied = _metadata_value(context, TOKEN_METADATA_KEY)
        if not expected or not hmac.compare_digest(expected, supplied):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid control-plane credentials")

    def ExecuteRun(self, request, context):
        """执行非流式 Agent 轮次并返回最终答案。"""
        self._authenticate(context)
        run_id = request.run_id.strip()
        message = request.message.strip()
        if (
            request.user_id <= 0
            or not run_id
            or len(run_id) > 64
            or not message
            or len(message) > 20_000
            or len(request.conversation_id) > 64
        ):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid run request")
        try:
            with session_scope():
                answer, conversation_id = run_agent(
                    message=message,
                    user_id=request.user_id,
                    conversation_id=request.conversation_id or None,
                    run_id=run_id,
                )
        except AgentRunConflictError:
            context.abort(grpc.StatusCode.ABORTED, "run already executing or not replayable")
        except Exception:
            LOGGER.exception("gRPC ExecuteRun failed", extra={"run_id": run_id})
            context.abort(grpc.StatusCode.INTERNAL, "agent execution failed")
        return agent_execution_pb2.ExecuteRunResponse(
            answer=answer,
            conversation_id=conversation_id,
            metadata_json=json.dumps(
                {"runtime": "python-langgraph-grpc"}, separators=(",", ":")
            ).encode("utf-8"),
        )

    def ExecuteRunStream(self, request, context):
        """把 Agent generator 事件转换为 server-streaming gRPC 消息。"""
        self._authenticate(context)
        run_id = request.run_id.strip()
        message = request.message.strip()
        if (
            request.user_id <= 0 or not run_id or len(run_id) > 64 or not message
            or len(message) > 20_000 or len(request.conversation_id) > 64
        ):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid run request")
        generator = None
        try:
            with session_scope():
                generator = run_agent_stream(
                    message=message,
                    user_id=request.user_id,
                    conversation_id=request.conversation_id or None,
                    run_id=run_id,
                    persist_runtime=False,
                    # The Go gateway creates runs directly in the control
                    # plane. The Agent appends only when the same user turn is
                    # not already the latest checkpoint message.
                    user_message_persisted=False,
                )
                for event in generator:
                    if not context.is_active():
                        generator.close()
                        return
                    event_type = str(event.get("type", ""))
                    yield agent_execution_pb2.RunEvent(
                        run_id=run_id,
                        sequence=int(event.get("sequence", 0)),
                        type=event_type,
                        payload_json=json.dumps(
                            event.get("payload") or {}, ensure_ascii=False, separators=(",", ":")
                        ).encode("utf-8"),
                        terminal=event_type in {"run.completed", "run.failed", "run.cancelled"},
                    )
        except GeneratorExit:
            if generator is not None:
                generator.close()
            raise
        except Exception:
            LOGGER.exception("gRPC ExecuteRunStream failed", extra={"run_id": run_id})
            context.abort(grpc.StatusCode.INTERNAL, "agent execution failed")

    def Check(self, request, context):
        """返回 serving，表示 gRPC 进程可以接收请求。"""
        self._authenticate(context)
        return agent_execution_pb2.HealthCheckResponse(status="serving")

    def ExecuteAction(self, request, context):
        """幂等执行已经由 Control Plane 批准的动作。"""
        self._authenticate(context)
        try:
            payload = json.loads(bytes(request.payload_json or b"{}").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload_json must contain valid JSON")
        if request.user_id <= 0 or not request.action_type or not isinstance(payload, dict):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid action request")
        try:
            with session_scope():
                proposal, created = create_action_proposal(
                    user_id=request.user_id,
                    action_type=request.action_type,
                    payload=payload,
                    reason=request.reason or "Go Control Plane approved action",
                    idempotency_key=request.idempotency_key,
                    reopen_terminal=False,
                )
                if created or proposal.status != "succeeded":
                    proposal = approve_and_execute_action(request.user_id, proposal.proposal_id)
                result = proposal.to_dict()
        except ActionConflictError:
            context.abort(grpc.StatusCode.ABORTED, "action already executing or not replayable")
        except ActionValidationError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception:
            LOGGER.exception(
                "gRPC ExecuteAction failed",
                extra={"action_type": request.action_type, "user_id": request.user_id},
            )
            context.abort(grpc.StatusCode.INTERNAL, "action execution failed")
        return agent_execution_pb2.ExecuteActionResponse(
            result_json=json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )

    def ValidateAction(self, request, context):
        """Validate and normalize a draft before the control plane stores it."""
        self._authenticate(context)
        try:
            payload = json.loads(bytes(request.payload_json or b"{}").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload_json must contain valid JSON")
        if request.user_id <= 0 or not request.action_type or not isinstance(payload, dict):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid action request")
        try:
            with session_scope():
                normalized = validate_action_payload(
                    request.user_id, request.action_type, payload
                )
                impact = describe_action_impact(request.action_type, normalized)
        except ActionValidationError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception:
            LOGGER.exception(
                "gRPC ValidateAction failed",
                extra={"action_type": request.action_type, "user_id": request.user_id},
            )
            context.abort(grpc.StatusCode.INTERNAL, "action validation failed")
        return agent_execution_pb2.ValidateActionResponse(
            payload_json=json.dumps(
                normalized, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"),
            impact=impact,
        )

    def DispatchCompute(self, request, context):
        """Dispatch an already-persisted job; Go owns validation and CRUD."""
        self._authenticate(context)
        if request.record_id <= 0 or not request.task_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid compute request")
        try:
            from app.utils.tasks import (
                run_agent_evaluation_task,
                run_nas_search_task,
                run_training_task,
                run_valuating_task,
            )

            try:
                payload = json.loads(bytes(request.payload_json or b"{}").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "payload_json must contain valid JSON",
                )
            if not isinstance(payload, dict):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "payload_json must contain an object",
                )

            if request.task_type == "agent_evaluation":
                raw_case_ids = payload.get("case_ids")
                if raw_case_ids is not None and (
                    not isinstance(raw_case_ids, list)
                    or any(not isinstance(item, str) for item in raw_case_ids)
                ):
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "case_ids must be a string array",
                    )
                run_agent_evaluation_task.apply_async(
                    args=[request.task_id, request.record_id, raw_case_ids],
                    task_id=f"agent-eval-{request.task_id}",
                )
                return agent_execution_pb2.DispatchComputeResponse(
                    task_id=request.task_id, status="accepted"
                )

            tasks = {
                "training": run_training_task,
                "validation": run_valuating_task,
                "nas": run_nas_search_task,
            }
            task = tasks.get(request.task_type)
            if task is None:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "unsupported task_type")
            task.apply_async(args=[request.record_id], task_id=request.task_id)
        except Exception:
            LOGGER.exception(
                "gRPC DispatchCompute failed",
                extra={"task_type": request.task_type, "task_id": request.task_id},
            )
            context.abort(grpc.StatusCode.UNAVAILABLE, "compute dispatch failed")
        return agent_execution_pb2.DispatchComputeResponse(
            task_id=request.task_id, status="accepted"
        )

    def Knowledge(self, request, context):
        """Expose the Python RAG engine to the Go HTTP gateway without Flask."""
        self._authenticate(context)
        try:
            payload = json.loads(bytes(request.payload_json or b"{}").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload_json must contain valid JSON")
        if not isinstance(payload, dict):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload_json must contain an object")
        try:
            from app.rag import get_knowledge_store

            store = get_knowledge_store()
            if request.operation == "status":
                result = store.status()
            elif request.operation == "search":
                items = store.search(
                    query=str(payload.get("query", "")),
                    top_k=payload.get("top_k", 6),
                    modality=str(payload.get("modality", "")),
                    operator=str(payload.get("operator", "")),
                    metadata_filters=payload.get("metadata_filters") or {},
                )
                assessment = store.assess(str(payload.get("query", "")), items)
                result = {
                    "items": [item.to_dict() for item in items],
                    "assessment": assessment.to_dict(),
                }
            elif request.operation == "rebuild":
                result = store.rebuild()
            else:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "unsupported knowledge operation")
        except Exception:
            LOGGER.exception("gRPC Knowledge failed", extra={"operation": request.operation})
            context.abort(grpc.StatusCode.INTERNAL, "knowledge operation failed")
        return agent_execution_pb2.KnowledgeResponse(
            result_json=json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )

    def RenderImage(self, request, context):
        """Render trusted dataset files for the Go API, including NumPy radar arrays."""
        self._authenticate(context)
        path = request.path.strip()
        if not path or not os.path.isfile(path):
            context.abort(grpc.StatusCode.NOT_FOUND, "image not found")
        try:
            from pathlib import Path

            from PIL import Image as PILImage

            source = Path(path)
            if source.suffix.casefold() == ".npy":
                import numpy as np

                from app.utils.utils import rd_to_image

                buffer = rd_to_image(np.load(source))
                content = buffer.getvalue()
            else:
                image = PILImage.open(source)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                content = buffer.getvalue()
            filename = source.stem + ".png"
        except Exception:
            LOGGER.exception("gRPC RenderImage failed", extra={"path": path})
            context.abort(grpc.StatusCode.INTERNAL, "image conversion failed")
        return agent_execution_pb2.RenderImageResponse(
            content=content, content_type="image/png", filename=filename
        )

    def CancelCompute(self, request, context):
        """请求 Celery 撤销或终止指定计算任务。"""
        self._authenticate(context)
        if not request.task_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "task_id is required")
        try:
            from app.utils.celery_worker import celery_app

            celery_app.control.revoke(request.task_id, terminate=request.terminate)
        except Exception:
            LOGGER.exception("gRPC CancelCompute failed", extra={"task_id": request.task_id})
            context.abort(grpc.StatusCode.UNAVAILABLE, "compute cancellation failed")
        return agent_execution_pb2.CancelComputeResponse(accepted=True)


# 初始化数据库并创建带线程池和消息大小限制的 gRPC Server。
def create_server() -> grpc.Server:
    init_database()
    max_workers = int(os.getenv("AGENT_GRPC_MAX_WORKERS", "8"))
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=(
            ("grpc.max_receive_message_length", 8 << 20),
            ("grpc.max_send_message_length", 32 << 20),
        ),
    )
    agent_execution_pb2_grpc.add_AgentExecutionServiceServicer_to_server(
        AgentExecutionServicer(), server
    )
    address = os.getenv("AGENT_GRPC_ADDRESS", "0.0.0.0:50051")
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"cannot bind gRPC server to {address}")
    return server


# 启动 gRPC Server，并在中断信号下执行宽限关闭。
def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    server = create_server()
    server.start()
    LOGGER.info("Python Agent gRPC server listening on %s", os.getenv("AGENT_GRPC_ADDRESS", "0.0.0.0:50051"))
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5).wait()


if __name__ == "__main__":
    main()
