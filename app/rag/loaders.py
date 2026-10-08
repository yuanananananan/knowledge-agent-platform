"""文件用途：读取支持的语料格式，并规范化为带来源信息的文档分段。"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class SourceSection:
    """统一表示从任意源文件读取出的一个带来源信息的章节。"""
    text: str
    section_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _markdown_sections(path: Path) -> list[SourceSection]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    frontmatter: dict[str, Any] = {}
    if raw_lines and raw_lines[0].strip() == "---":
        try:
            end = raw_lines[1:].index("---") + 1
            import yaml
            frontmatter = yaml.safe_load("\n".join(raw_lines[1:end])) or {}
            raw_lines = raw_lines[end + 1:]
        except (ValueError, TypeError):
            frontmatter = {}
    stack: list[str] = []
    current_path: list[str] = []
    body: list[str] = []
    sections: list[SourceSection] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append(SourceSection(text=text, section_path=list(current_path), metadata=dict(frontmatter)))
        body.clear()

    for line in raw_lines:
        match = HEADING.match(line)
        if not match:
            body.append(line)
            continue
        flush()
        level = len(match.group(1))
        title = match.group(2).strip()
        stack[level - 1:] = [title]
        current_path = list(stack)
    flush()
    return sections


def _json_sections(path: Path) -> list[SourceSection]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("operators"), list):
        return [
            SourceSection(
                text=json.dumps(item, ensure_ascii=False, indent=2),
                section_path=["operators", str(item.get("name", item.get("id", "unknown")))],
                metadata={
                    key: item[key]
                    for key in ("modality", "category", "display_name", "summary", "aliases", "supported_claims")
                    if key in item
                } | ({"entity_id": item["name"]} if item.get("name") else {}),
            )
            for item in data["operators"]
        ]
    items = data if isinstance(data, list) else [data]
    return [
        SourceSection(
            text=json.dumps(item, ensure_ascii=False, indent=2),
            section_path=[f"item-{index + 1}"],
        )
        for index, item in enumerate(items)
    ]


def _csv_sections(path: Path) -> list[SourceSection]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        sections = []
        for index, row in enumerate(reader):
            text = "\n".join(f"{key}: {value}" for key, value in row.items())
            operator = row.get("operator") or row.get("name") or f"row-{index + 1}"
            sections.append(SourceSection(
                text=text,
                section_path=[operator],
                metadata={
                    "modality": row.get("modality"),
                    "operator": operator,
                },
            ))
        return sections


def _pdf_sections(path: Path) -> list[SourceSection]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("解析 PDF 需要安装 pypdf，请执行 pip install -r requirements.txt") from exc
    reader = PdfReader(str(path))
    ocr_enabled = os.getenv("RAG_OCR_ENABLED", "true").casefold() in {"1", "true", "yes", "on"}
    minimum_text = max(0, int(os.getenv("RAG_OCR_MIN_TEXT_CHARS", "40")))
    pdfium_document = None
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        text_layer = (page.extract_text() or "").strip()
        text = text_layer
        metadata: dict[str, Any] = {
            "extraction_method": "text_layer",
            "text_layer_characters": len(re.sub(r"\s+", "", text_layer)),
        }
        if ocr_enabled and metadata["text_layer_characters"] < minimum_text:
            try:
                if pdfium_document is None:
                    import pypdfium2 as pdfium
                    pdfium_document = pdfium.PdfDocument(str(path))
                ocr = _ocr_pdf_page(pdfium_document, page_number - 1)
                if len(re.sub(r"\s+", "", ocr["text"])) > metadata["text_layer_characters"]:
                    text = ocr["text"]
                    metadata.update({
                        "extraction_method": "ocr",
                        "ocr_engine": "tesseract",
                        "ocr_language": ocr["language"],
                        "ocr_confidence": ocr["confidence"],
                        "ocr_language_fallback": ocr["language_fallback"],
                    })
            except Exception as exc:
                metadata["ocr_error"] = str(exc)[:500]
        if text:
            sections.append(SourceSection(
                text=text,
                section_path=[f"第{page_number}页"],
                page_start=page_number,
                page_end=page_number,
                metadata=metadata,
            ))
    if not sections:
        raise ValueError(f"PDF 未提取到文本，可能需要 OCR: {path.name}")
    return sections


@lru_cache(maxsize=1)
def _tesseract_languages() -> set[str]:
    executable = shutil.which(os.getenv("RAG_TESSERACT_COMMAND", "tesseract"))
    if not executable:
        raise RuntimeError("未找到 tesseract 命令")
    completed = subprocess.run(
        [executable, "--list-langs"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return {
        line.strip() for line in completed.stdout.splitlines()
        if line.strip() and not line.startswith("List of available")
    }


def _choose_ocr_language() -> tuple[str, bool]:
    requested = [item.strip() for item in os.getenv("RAG_OCR_LANGUAGES", "chi_sim+eng").split("+") if item.strip()]
    available = _tesseract_languages()
    selected = [item for item in requested if item in available]
    if not selected:
        if "eng" in available:
            return "eng", True
        if available:
            return sorted(available)[0], True
        raise RuntimeError("Tesseract 没有可用语言包")
    return "+".join(selected), len(selected) != len(requested)


def _parse_tesseract_tsv(tsv: str) -> tuple[str, float | None]:
    lines: dict[tuple[str, str, str, str], list[str]] = {}
    confidences: list[float] = []
    rows = tsv.splitlines()
    if not rows:
        return "", None
    headers = rows[0].split("\t")
    positions = {name: index for index, name in enumerate(headers)}
    for raw in rows[1:]:
        values = raw.split("\t")
        if len(values) < len(headers):
            continue
        word = values[positions.get("text", len(values) - 1)].strip()
        if not word:
            continue
        key = tuple(values[positions[name]] for name in ("block_num", "par_num", "line_num", "page_num"))
        lines.setdefault(key, []).append(word)
        try:
            confidence = float(values[positions["conf"]])
            if confidence >= 0:
                confidences.append(confidence)
        except (KeyError, ValueError):
            pass
    text = "\n".join(" ".join(words) for words in lines.values()).strip()
    confidence = round(sum(confidences) / len(confidences), 2) if confidences else None
    return text, confidence


def _ocr_pdf_page(pdfium_document, page_index: int) -> dict[str, Any]:
    try:
        from PIL import ImageOps
    except ImportError as exc:
        raise RuntimeError("OCR 图像预处理需要 Pillow") from exc
    executable = shutil.which(os.getenv("RAG_TESSERACT_COMMAND", "tesseract"))
    if not executable:
        raise RuntimeError("未找到 tesseract 命令")
    language, fallback = _choose_ocr_language()
    dpi = max(96, min(int(os.getenv("RAG_OCR_DPI", "240")), 400))
    timeout = max(10, int(os.getenv("RAG_OCR_TIMEOUT_SECONDS", "90")))
    image = pdfium_document[page_index].render(scale=dpi / 72).to_pil()
    image = ImageOps.autocontrast(ImageOps.grayscale(image))
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary:
            temporary_path = temporary.name
        image.save(temporary_path, format="PNG")
        completed = subprocess.run(
            [executable, temporary_path, "stdout", "-l", language, "--psm", "6", "tsv"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        text, confidence = _parse_tesseract_tsv(completed.stdout)
        return {
            "text": text,
            "confidence": confidence,
            "language": language,
            "language_fallback": fallback,
        }
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


def load_sections(path: Path) -> list[SourceSection]:
    """根据文件扩展名选择解析器，将文件加载为标准章节列表。"""
    suffix = path.suffix.casefold()
    if suffix == ".md":
        return _markdown_sections(path)
    if suffix == ".json":
        return _json_sections(path)
    if suffix == ".csv":
        return _csv_sections(path)
    if suffix == ".pdf":
        return _pdf_sections(path)
    raise ValueError(f"不支持的知识文件类型: {path.suffix}")


def split_section(section: SourceSection, max_chars: int = 1600, overlap_chars: int = 180) -> list[SourceSection]:
    """把过长章节切成有重叠的检索块，同时保留原始来源元数据。"""
    text = section.text.strip()
    if len(text) <= max_chars:
        return [section]
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            step = max(1, max_chars - overlap_chars)
            chunks.extend(paragraph[start:start + max_chars] for start in range(0, len(paragraph), step))
            continue
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            prefix = current[-overlap_chars:] if overlap_chars else ""
            current = f"{prefix}\n\n{paragraph}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [
        SourceSection(
            text=chunk,
            section_path=section.section_path,
            page_start=section.page_start,
            page_end=section.page_end,
            metadata=section.metadata,
        )
        for chunk in chunks
    ]
