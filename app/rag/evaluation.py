"""文件用途：针对 RAG Golden Dataset 离线评估召回、排序和回答充分性。"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.rag.store import HybridKnowledgeStore


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def load_golden_queries(path: Path) -> list[dict[str, Any]]:
    """逐行读取 JSONL 格式的 RAG 标准评测问题。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_retrieval(
    store: HybridKnowledgeStore,
    dataset_path: Path,
    top_k: int = 5,
    search_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """计算召回率、MRR、nDCG、证据覆盖率和延迟等检索指标。"""
    cases = load_golden_queries(dataset_path)
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    evidence_recalls: list[float] = []
    all_evidence_hits: list[float] = []
    ndcgs: list[float] = []
    claim_recalls: list[float] = []
    latencies: list[float] = []
    context_characters: list[int] = []
    details = []
    categories: dict[str, list[float]] = defaultdict(list)
    category_rows: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    difficulty_rows: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for case in cases:
        started = time.perf_counter()
        results = store.search(
            case["question"],
            top_k=top_k,
            include_superseded="version" in case["category"] or "conflict" in case["category"],
            metadata_filters=case.get("metadata_filters"),
            **(search_options or {}),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        context_characters.append(sum(len(item.content) for item in results))
        expected = {Path(item).name for item in case.get("expected_sources", [])}
        returned = [Path(item.document_path).name for item in results]
        required_claims = set(case.get("required_claims", []))
        returned_claims = {claim for item in results for claim in item.claim_ids}
        claim_recall = len(required_claims & returned_claims) / len(required_claims) if required_claims else None
        ranks = [index + 1 for index, item in enumerate(results)
                 if (required_claims & set(item.claim_ids)) or (Path(item.document_path).name in expected)]
        hit = float(bool(ranks))
        rr = 1.0 / min(ranks) if ranks else 0.0
        evidence_recall = len(expected.intersection(returned)) / len(expected) if expected else 1.0
        all_evidence_hit = float(evidence_recall == 1.0)
        relevance = {
            Path(name).name: int(grade)
            for name, grade in case.get("relevance", {}).items()
        } or {name: 1 for name in expected}
        seen_relevant_sources: set[str] = set()
        dcg = 0.0
        for rank, name in enumerate(returned, start=1):
            # Multiple chunks from one source are useful context, but the source
            # relevance judgment may contribute to nDCG only once.
            grade = 0 if name in seen_relevant_sources else relevance.get(name, 0)
            if grade:
                seen_relevant_sources.add(name)
            dcg += (2 ** grade - 1) / math.log2(rank + 1)
        ideal = sorted(relevance.values(), reverse=True)[:top_k]
        idcg = sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
        ndcg = dcg / idcg if idcg else 1.0
        recalls.append(hit)
        evidence_recalls.append(evidence_recall)
        all_evidence_hits.append(all_evidence_hit)
        ndcgs.append(ndcg)
        if claim_recall is not None:
            claim_recalls.append(claim_recall)
        reciprocal_ranks.append(rr)
        categories[case["category"]].append(hit)
        metric_row = (hit, rr, evidence_recall, ndcg)
        category_rows[case["category"]].append(metric_row)
        difficulty_rows[str(case.get("difficulty", "unspecified"))].append(metric_row)
        details.append({
            "id": case["id"],
            "category": case["category"],
            "hit": bool(hit),
            "reciprocal_rank": rr,
            "evidence_recall": round(evidence_recall, 4),
            "all_evidence_hit": bool(all_evidence_hit),
            "ndcg": round(ndcg, 4),
            "claim_recall": round(claim_recall, 4) if claim_recall is not None else None,
            "required_claims": sorted(required_claims),
            "returned_claims": sorted(returned_claims),
            "latency_ms": round(latency_ms, 2),
            "expected_sources": sorted(expected),
            "returned_sources": returned,
        })
    return {
        "cases": len(cases),
        f"recall_at_{top_k}": statistics.mean(recalls) if recalls else 0.0,
        "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        f"evidence_recall_at_{top_k}": statistics.mean(evidence_recalls) if evidence_recalls else 0.0,
        f"all_evidence_recall_at_{top_k}": statistics.mean(all_evidence_hits) if all_evidence_hits else 0.0,
        f"ndcg_at_{top_k}": statistics.mean(ndcgs) if ndcgs else 0.0,
        f"claim_recall_at_{top_k}": statistics.mean(claim_recalls) if claim_recalls else None,
        "average_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": round(_percentile(latencies, 0.50), 2),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 2),
        "p99_latency_ms": round(_percentile(latencies, 0.99), 2),
        "average_context_characters": round(statistics.mean(context_characters), 2) if context_characters else 0.0,
        "category_recall": {
            key: round(statistics.mean(values), 4) for key, values in sorted(categories.items())
        },
        "category_metrics": {
            key: {
                "cases": len(values),
                f"recall_at_{top_k}": round(statistics.mean(item[0] for item in values), 4),
                "mrr": round(statistics.mean(item[1] for item in values), 4),
                f"evidence_recall_at_{top_k}": round(statistics.mean(item[2] for item in values), 4),
                f"ndcg_at_{top_k}": round(statistics.mean(item[3] for item in values), 4),
            }
            for key, values in sorted(category_rows.items())
        },
        "difficulty_metrics": {
            key: {
                "cases": len(values),
                f"recall_at_{top_k}": round(statistics.mean(item[0] for item in values), 4),
                "mrr": round(statistics.mean(item[1] for item in values), 4),
                f"evidence_recall_at_{top_k}": round(statistics.mean(item[2] for item in values), 4),
                f"ndcg_at_{top_k}": round(statistics.mean(item[3] for item in values), 4),
            }
            for key, values in sorted(difficulty_rows.items())
        },
        "details": details,
    }


def _call_generation_model(question: str, evidence: list[dict[str, Any]], assessment: dict[str, Any]) -> str:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("端到端生成评测需要 LLM_API_KEY")
    prompt = (
        "你是雷达实验知识助手。只能根据证据回答。证据不足时明确拒绝给出确定结论。"
        "每个知识结论后使用[来源：document]标注实际文档路径，不得编造来源。\n"
        f"证据状态：{json.dumps(assessment, ensure_ascii=False)}\n"
        f"问题：{question}\n证据：{json.dumps(evidence, ensure_ascii=False)}"
    )
    request = urllib.request.Request(
        f"{os.getenv('LLM_BASE_URL', 'https://api.deepseek.com').rstrip('/')}/chat/completions",
        data=json.dumps({
            "model": os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["choices"][0]["message"]["content"]).strip()


def evaluate_generation(store: HybridKnowledgeStore, dataset_path: Path, top_k: int = 5) -> dict[str, Any]:
    """运行检索与生成，并评估要点覆盖、引用有效性和拒答行为。"""
    cases = load_golden_queries(dataset_path)
    details = []
    coverage_scores, citation_scores, refusal_scores, latencies = [], [], [], []
    refusal_words = ("无法", "不能确定", "证据不足", "无法保证", "需要查询", "需要实验")
    for case in cases:
        started = time.perf_counter()
        results = store.search(
            case["question"], top_k=top_k,
            include_superseded="version" in case["category"] or "conflict" in case["category"],
            metadata_filters=case.get("metadata_filters"),
        )
        assessment = store.assess(case["question"], results).to_dict()
        evidence = [{"document": item.document_path, "content": item.content} for item in results]
        answer = _call_generation_model(case["question"], evidence, assessment)
        latency = (time.perf_counter() - started) * 1000
        normalized = answer.casefold()
        required = [str(point).casefold() for point in case.get("required_points", [])]
        coverage = sum(point in normalized for point in required) / len(required) if required else 1.0
        cited = re.findall(r"\[来源：([^\]]+)\]", answer)
        returned_paths = {item.document_path for item in results}
        citation_validity = (
            sum(any(path in citation for path in returned_paths) for citation in cited) / len(cited)
            if cited else 0.0
        )
        refused = any(word in answer for word in refusal_words)
        refusal_correct = float(refused == bool(case.get("should_refuse", False)))
        coverage_scores.append(coverage)
        citation_scores.append(citation_validity)
        refusal_scores.append(refusal_correct)
        latencies.append(latency)
        details.append({
            "id": case["id"], "answer": answer, "assessment": assessment,
            "required_point_coverage": round(coverage, 4),
            "citation_validity": round(citation_validity, 4),
            "refusal_correct": bool(refusal_correct), "latency_ms": round(latency, 2),
        })
    return {
        "cases": len(cases),
        "required_point_coverage": statistics.mean(coverage_scores) if coverage_scores else 0.0,
        "citation_validity": statistics.mean(citation_scores) if citation_scores else 0.0,
        "refusal_accuracy": statistics.mean(refusal_scores) if refusal_scores else 0.0,
        "average_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(_percentile(latencies, 0.95), 2),
        "details": details,
        "metric_limitations": "字符串覆盖和引用来源有效性是自动代理指标；严格忠实度仍需人工或独立评审模型逐句判断。",
    }
