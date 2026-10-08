"""文件用途：解析 NAS 搜索日志中的 epoch、结构、指标和最佳结果。"""

import re
from turtledemo.chaos import line
from typing import Any, Dict, List, Optional

from NAS.genotypes import Genotype

_VALID_ACC_RE = re.compile(r"valid_acc\s+([0-9]+(?:\.[0-9]+)?)")
_GENOTYPE_RE = re.compile(r"genotype\s*=\s*(Genotype\(.*\))")
_RANGE_RE = re.compile(r"range\((\d+)\s*,\s*(\d+)\)")

# range(2, 6)
# 转换成：
# [2, 3, 4, 5]
def _range_to_list(match: re.Match) -> str:
    start = int(match.group(1))
    end = int(match.group(2))
    return str(list(range(start, end)))

#把 NAS 日志里的 Genotype(...) 字符串，转换成 Python 字典。
def genotype_str_to_dict(genotype_text: str) -> Optional[Dict[str, Any]]:
    if not genotype_text:
        return None

    text = genotype_text.strip()

    if text.startswith("genotype ="):
        text = text.split("=", 1)[1].strip()

    match = _GENOTYPE_RE.search(text)
    if match:
        text = match.group(1)

    if not(text.startswith("Genotype(") and text.endswith(")")):
        return None

    inner = text[len("Genotype("):-1]
    inner = _RANGE_RE.sub(_range_to_list, inner)

    safe_expr = f"dict({inner})"

    try:
        data = eval(safe_expr, {"__builtins__": {}}, {"dict": dict})
    except Exception:
        return None

    def to_jsonable(obj):
        if isinstance(obj, tuple):
            return [to_jsonable(x) for x in obj]
        if isinstance(obj, list):
            return [to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        return obj

    return to_jsonable(data)

#从日志中找到 NAS 搜索输出目录
#比如日志中有一行：
#Experiment dir : model_ckp/search-EXP-sar0612-185011
#返回：
#"model_ckp/search-EXP-sar0612-185011“
def extract_save_dir(line: str) -> Optional[str]:
    marker = "Experiment dir :"
    if marker in line:
        return line.split(marker, 1)[1].strip()
    return None

#从完整的日志中找valid_acc最高的一轮genotype
def parse_best_genotype_from_lines(lines: List[str]) -> Dict[str, Any]:
    last_genotype_text = None
    best_acc = None
    best_genotype_text = None

    for line in lines:
        genotype_match = _GENOTYPE_RE.search(line)
        if genotype_match:
            last_genotype_text = genotype_match.group(1)
            continue

        acc_match = _VALID_ACC_RE.search(line)
        if acc_match and last_genotype_text:
            acc = float(acc_match.group(1))
            if best_acc is None or acc > best_acc:
                best_acc = acc
                best_genotype_text = last_genotype_text

    best_genotype = genotype_str_to_dict(best_genotype_text) if best_genotype_text else None

    return{
        "best_valid_acc": best_acc,
        "best_genotype": best_genotype,
        "best_genotype_text": best_genotype_text,
    }
