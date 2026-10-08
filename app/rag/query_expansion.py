"""文件用途：为 RAG-Fusion 生成领域中立的多查询扩展并合并召回结果。"""
from __future__ import annotations
import json
import os
import re
import urllib.request

def _unique(items: list[str], limit: int) -> list[str]:
    output, seen = [], set()
    for item in items:
        value = re.sub(r"\s+", " ", str(item)).strip(); key = value.casefold()
        if value and key not in seen: seen.add(key); output.append(value)
        if len(output) >= limit: break
    return output

class QueryExpander:
    """查询扩展器的统一接口。"""
    name = "base"
    def expand(self, query: str, max_queries: int = 4) -> list[str]: raise NotImplementedError

class IdentityQueryExpander(QueryExpander):
    """不改写问题，作为关闭扩展或故障降级时的实现。"""
    name = "identity-v1"
    def expand(self, query: str, max_queries: int = 4) -> list[str]: return [query]

class DomainQueryExpander(IdentityQueryExpander):
    """保留旧配置名称的兼容实现，本身不注入任何领域词汇。"""
    name = "metadata-driven-v2"

class LLMQueryExpander(QueryExpander):
    """调用大模型生成语义等价但措辞不同的检索问题。"""
    name = "llm-domain-neutral-v2"
    def __init__(self):
        self.api_key = os.getenv("RAG_QUERY_EXPANSION_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("RAG_QUERY_EXPANSION_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("RAG_QUERY_EXPANSION_MODEL") or os.getenv("LLM_MODEL_NAME", "deepseek-chat")
        self.timeout = float(os.getenv("RAG_QUERY_EXPANSION_TIMEOUT_SECONDS", "20"))
        if not self.api_key: raise RuntimeError("LLM 查询扩展缺少 API Key")
    def expand(self, query: str, max_queries: int = 4) -> list[str]:
        prompt = ("Rewrite the search request into semantically distinct but equivalent retrieval queries. "
                  "Preserve every identifier, number, named entity, negation and constraint. Do not answer, "
                  "infer domain facts, or invent entity names. Return only a JSON string array. "
                  f"Generate at most {max_queries - 1} rewrites. Request: {query}")
        request = urllib.request.Request(f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps({"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response: payload = json.loads(response.read().decode())
        content = payload["choices"][0]["message"]["content"]; match = re.search(r"\[[\s\S]*\]", content)
        return _unique([query, *(json.loads(match.group(0)) if match else [])], max_queries)

class FallbackQueryExpander(QueryExpander):
    """主扩展器失败时自动退回到安全的基础实现。"""
    def __init__(self, primary: QueryExpander, fallback: QueryExpander):
        self.primary, self.fallback = primary, fallback; self.name = f"{primary.name}|fallback:{fallback.name}"
    def expand(self, query: str, max_queries: int = 4) -> list[str]:
        try: return self.primary.expand(query, max_queries)
        except Exception: return self.fallback.expand(query, max_queries)

def build_query_expander() -> QueryExpander:
    """根据配置创建查询扩展器，并为远程模型配置降级路径。"""
    fallback = IdentityQueryExpander(); provider = os.getenv("RAG_QUERY_EXPANSION_PROVIDER", "llm").casefold()
    if provider in {"llm", "model"}:
        try: return FallbackQueryExpander(LLMQueryExpander(), fallback)
        except Exception: return fallback
    return fallback
