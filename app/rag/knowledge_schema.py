"""文件用途：从知识文档中提取通用实体、声明和文档元数据。"""
from __future__ import annotations
import hashlib
import re
from typing import Any

def normalize_list(value: Any) -> list[str]:
    """把空值、单值或多种分隔格式统一整理为字符串列表。"""
    if value is None: return []
    if isinstance(value, list): return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in re.split(r"[,;|]", str(value)) if x.strip()]

def extract_entities(metadata: dict[str, Any], title: str, text: str) -> list[dict[str, str]]:
    """从文档元数据中提取可建立索引的实体、别名和简介。"""
    explicit = normalize_list(metadata.get("entities"))
    explicit.extend(normalize_list(metadata.get("operators")))
    for key in ("entity_id", "operator"):
        if metadata.get(key): explicit.append(str(metadata[key]))
    aliases = normalize_list(metadata.get("aliases"))
    if metadata.get("display_name"): aliases.append(str(metadata["display_name"]))
    summary = str(metadata.get("summary") or text[:800])
    return [{"entity_id": entity, "aliases": " | ".join(dict.fromkeys([entity, *aliases])),
             "profile": f"{entity}\n{title}\n{summary}"} for entity in dict.fromkeys(explicit)]

def extract_claims(metadata: dict[str, Any], entity_ids: list[str], section_path: list[str], text: str) -> list[dict[str, str]]:
    """提取显式声明；未提供声明时按章节生成稳定的声明标识。"""
    explicit = normalize_list(metadata.get("supported_claims") or metadata.get("claims"))
    if explicit: return [{"claim_id": claim, "claim_text": text} for claim in explicit]
    heading = " > ".join(section_path).strip() or "document"
    subject = entity_ids[0] if len(entity_ids) == 1 else str(metadata.get("document_id") or metadata.get("path") or "document")
    slug = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "-", heading.casefold()).strip("-")[:80]
    return [{"claim_id": f"{subject}.section.{slug or hashlib.sha1(heading.encode()).hexdigest()[:10]}", "claim_text": text}]

def evidence_role(metadata: dict[str, Any]) -> str:
    """返回文档在证据体系中的标准化角色。"""
    return str(metadata.get("evidence_role") or metadata.get("document_role") or "reference").casefold()
