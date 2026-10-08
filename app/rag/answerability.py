"""文件用途：根据检索证据评估问题是否可回答，并给出拒答或澄清建议。"""
from __future__ import annotations
import json
import os
import re
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

SUPPORTED = "SUPPORTED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
CONTRADICTED = "CONTRADICTED"
VALID_STATES = {SUPPORTED, INSUFFICIENT_EVIDENCE, OUT_OF_SCOPE, CONTRADICTED}

@dataclass
class EvidenceAssessment:
    """描述检索证据能否支持回答，以及下一步应该回答、澄清还是拒答。"""
    state: str
    reason: str
    next_action: str
    supported_claims: list[str] | None = None
    unsupported_claims: list[str] | None = None
    def to_dict(self) -> dict[str, Any]: return asdict(self)

def assess_evidence(query: str, results: list[Any]) -> EvidenceAssessment:
    """在不调用模型时，用保守规则判断检索证据是否足够。"""
    if not results:
        return EvidenceAssessment(OUT_OF_SCOPE, "没有找到可用知识证据", "补充资料或澄清问题", [], [query])
    best = results[0]
    if best.keyword_score <= 0 and best.vector_score < float(os.getenv("RAG_OUT_OF_SCOPE_VECTOR_THRESHOLD", "0.25")):
        return EvidenceAssessment(OUT_OF_SCOPE, "最高候选的词法和语义相关性均较低", "补充资料或澄清问题", [], [query])
    return EvidenceAssessment(SUPPORTED, "已找到相关证据；模型不可用，未执行蕴含判断", "基于证据回答并标注来源", [], [])

class LLMEvidenceAssessor:
    """调用大模型对问题和证据做蕴含关系判定。"""
    def __init__(self):
        self.api_key = os.getenv("RAG_EVIDENCE_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("RAG_EVIDENCE_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("RAG_EVIDENCE_MODEL") or os.getenv("LLM_MODEL_NAME", "deepseek-chat")
        if not self.api_key: raise RuntimeError("证据判定缺少API Key")
    def assess(self, query: str, results: list[Any]) -> EvidenceAssessment:
        evidence = [{"source": x.document_path, "text": x.matched_content} for x in results]
        prompt = ("Judge whether the supplied evidence entails the answer requested by the user. Return JSON only with "
                  "state (SUPPORTED, INSUFFICIENT_EVIDENCE, CONTRADICTED, or OUT_OF_SCOPE), reason, next_action, "
                  "supported_claims, and unsupported_claims. Topic similarity alone is not entailment. Do not use "
                  "outside knowledge.\nQuestion: " + query + "\nEvidence: " + json.dumps(evidence, ensure_ascii=False))
        request = urllib.request.Request(f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps({"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=float(os.getenv("RAG_EVIDENCE_TIMEOUT_SECONDS", "30"))) as response:
            content = json.loads(response.read().decode())["choices"][0]["message"]["content"]
        match = re.search(r"\{[\s\S]*\}", content); data = json.loads(match.group(0)) if match else {}
        state = str(data.get("state", "")).upper()
        if state not in VALID_STATES: raise ValueError("无效证据状态")
        return EvidenceAssessment(state, str(data.get("reason", "")), str(data.get("next_action", "")),
                                  list(data.get("supported_claims", [])), list(data.get("unsupported_claims", [])))

def build_evidence_assessor():
    """根据环境变量创建证据判定器；配置不可用时返回空。"""
    if os.getenv("RAG_EVIDENCE_ASSESSOR", "llm").casefold() == "llm":
        try: return LLMEvidenceAssessor()
        except Exception: pass
    return None
