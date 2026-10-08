"""文件用途：把自然语言问题解析为可审计的结构化检索计划。"""
from __future__ import annotations

import json
import os
import re
import unicodedata
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryPlan:
    """保存标准化问题、子查询、实体和必需证据的结构化检索计划。"""
    original: str
    normalized: str
    subqueries: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    required_roles: list[str] = field(default_factory=list)
    required_claims: list[str] = field(default_factory=list)
    complex_query: bool = False
    planner: str = "heuristic"


@dataclass
class QueryRoute:
    """描述问题复杂度以及是否需要模型规划和查询扩展。"""
    kind: str
    use_model_planner: bool
    use_query_expansion: bool
    reason: str


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(query)).strip())


def route_query(query: str, linked_entities: list[dict] | None = None) -> QueryRoute:
    """根据问题结构和已链接实体选择合适的检索成本。"""
    normalized = normalize_query(query)
    lowered = normalized.casefold()
    linked = linked_entities or []
    entities = {str(item.get("entity_id", "")) for item in linked if item.get("entity_id")}
    exact_entities = {
        str(item.get("entity_id", "")) for item in linked if item.get("match") == "exact"
    }
    comparison = any(x in lowered for x in ("比较", "对比", "区别", "差异", "compare", "versus", " vs "))
    procedure = any(x in lowered for x in ("如何", "怎么", "步骤", "方案", "设计", "流程", "how ", "design", "procedure"))
    version = any(x in lowered for x in ("版本", "旧版", "新版", "变更", "迁移", "冲突", "version", "migration"))
    multi_part = normalized.count("？") + normalized.count("?") + normalized.count("；") + normalized.count(";") >= 2
    constrained = len(normalized) >= int(os.getenv("RAG_COMPLEX_QUERY_MIN_CHARS", "80")) and any(
        x in lowered for x in ("并且", "同时", "以及", "分别", "还要", "and", "also")
    )
    complex_query = len(entities) > 1 or comparison or procedure or version or multi_part or constrained
    if complex_query:
        return QueryRoute(
            "complex", True, not bool(exact_entities),
            "multi-entity or structural comparison/procedure/version request",
        )
    if exact_entities:
        return QueryRoute("explicit_fact", False, False, "single explicit indexed entity")
    return QueryRoute("fuzzy", False, True, "no explicit indexed entity")


def _balanced_claims(claims: list[str], limit: int) -> list[str]:
    """Round-robin claim subjects so one entity cannot consume the plan."""
    groups: dict[str, list[str]] = {}
    for claim in claims:
        parts = claim.split(".")
        subject = ".".join(parts[:2]) if len(parts) >= 3 else parts[0]
        groups.setdefault(subject, []).append(claim)
    output: list[str] = []
    while len(output) < limit and any(groups.values()):
        for values in groups.values():
            if values and len(output) < limit:
                output.append(values.pop(0))
    return output


def build_query_plan(query: str, linked_entities: list[dict] | None = None) -> QueryPlan:
    """模型规划器不可用时，生成领域中立的规则检索计划。"""
    normalized = normalize_query(query)
    lowered = normalized.casefold()
    entities = [str(item["entity_id"]) for item in (linked_entities or [])]
    compare = any(x in lowered for x in ("比较", "对比", "区别", "差异", "compare", "versus", " vs "))
    procedure = any(x in lowered for x in ("如何", "怎么", "步骤", "方案", "设计", "流程", "how ", "design", "procedure"))
    version = any(x in lowered for x in ("版本", "旧版", "新版", "变更", "冲突", "version"))
    subqueries = [f"{entity} definition applicability limitations" for entity in entities]
    roles: list[str] = []
    if compare:
        subqueries.append("comparison differences applicability limitations"); roles.append("comparison")
    if procedure:
        subqueries.append("procedure requirements controls validation"); roles.append("procedure")
    if version:
        subqueries.append("version change superseded current conflict"); roles.append("version")
    return QueryPlan(query, normalized, list(dict.fromkeys(subqueries)), list(dict.fromkeys(entities)),
                     list(dict.fromkeys(roles)), [], len(entities) > 1 or bool(compare or procedure or version))


class QueryPlanner:
    """基于通用规则生成检索计划的基础规划器。"""
    name = "heuristic-domain-neutral-v2"

    def plan(self, query: str, linked_entities: list[dict], context: dict[str, Any]) -> QueryPlan:
        return build_query_plan(query, linked_entities)


class LLMQueryPlanner(QueryPlanner):
    """让大模型在现有索引 Schema 范围内生成结构化检索计划。"""
    name = "llm-structured-grounded-v1"

    def __init__(self):
        self.api_key = os.getenv("RAG_QUERY_PLANNER_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("RAG_QUERY_PLANNER_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("RAG_QUERY_PLANNER_MODEL") or os.getenv("LLM_MODEL_NAME", "deepseek-chat")
        self.timeout = float(os.getenv("RAG_QUERY_PLANNER_TIMEOUT_SECONDS", "30"))
        if not self.api_key:
            raise RuntimeError("模型查询规划缺少API Key")

    def plan(self, query: str, linked_entities: list[dict], context: dict[str, Any]) -> QueryPlan:
        normalized = normalize_query(query)
        prompt = (
            "Plan retrieval using only the supplied index schema. Return JSON only with entities, required_claims, "
            "required_roles, subqueries, and complex_query. Entity, claim and role values must be copied exactly from "
            "the schema; never invent IDs. Split comparison, procedure, version and multi-part questions into "
            "self-contained subqueries. Preserve identifiers, numbers, negations and constraints. At most 5 "
            "subqueries and 10 minimal required claims. Do not answer the question.\nQuestion: " + normalized +
            "\nIndex schema: " + json.dumps(context, ensure_ascii=False)
        )
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps({"model": self.model, "temperature": 0,
                             "messages": [{"role": "user", "content": prompt}]}).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            content = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
        match = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(match.group(0)) if match else {}
        entity_ids = {item["entity_id"] for item in context.get("entities", [])}
        claim_ids = set(context.get("claims", []))
        entities = [str(x) for x in data.get("entities", []) if str(x) in entity_ids]
        entities = list(dict.fromkeys([*[str(x["entity_id"]) for x in linked_entities], *entities]))
        claims = _balanced_claims(
            list(dict.fromkeys(str(x) for x in data.get("required_claims", []) if str(x) in claim_ids)), 10
        )
        available_roles = set(context.get("roles", []))
        roles = list(dict.fromkeys(str(x).casefold() for x in data.get("required_roles", [])
                                   if str(x).casefold() in available_roles))[:5]
        subqueries = list(dict.fromkeys(normalize_query(x) for x in data.get("subqueries", []) if str(x).strip()))[:5]
        return QueryPlan(query, normalized, subqueries, entities, roles, claims,
                         bool(data.get("complex_query") or len(subqueries) > 1 or len(entities) > 1), self.name)


class FallbackQueryPlanner(QueryPlanner):
    """模型规划失败时回退到规则规划器。"""
    def __init__(self, primary: QueryPlanner, fallback: QueryPlanner):
        self.primary, self.fallback = primary, fallback
        self.name = f"{primary.name}|fallback:{fallback.name}"

    def plan(self, query: str, linked_entities: list[dict], context: dict[str, Any]) -> QueryPlan:
        try:
            return self.primary.plan(query, linked_entities, context)
        except Exception:
            return self.fallback.plan(query, linked_entities, context)


def build_query_planner() -> QueryPlanner:
    """根据配置组装模型规划器和规则降级方案。"""
    fallback = QueryPlanner()
    if os.getenv("RAG_QUERY_PLANNER_PROVIDER", "llm").casefold() in {"llm", "model"}:
        try:
            return FallbackQueryPlanner(LLMQueryPlanner(), fallback)
        except Exception:
            pass
    return fallback
