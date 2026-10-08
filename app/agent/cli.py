"""文件用途：提供 RAG 索引维护和 Agent 评测任务的独立命令行入口。"""

import json
from pathlib import Path

import click

from app import init_database, session_scope
from app.agent.evaluation import run_golden_evaluation, seed_golden_cases
from app.rag import get_knowledge_store
from app.rag.evaluation import evaluate_generation, evaluate_retrieval


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@click.group()
def cli():
    """维护知识索引和 Agent 评测数据集的顶层命令组。"""
    init_database()


@cli.group("rag")
def rag():
    """构建、检查和评测混合 RAG 索引。"""


@rag.command("index")
def rag_index():
    """从当前知识语料完整重建索引。"""
    click.echo(json.dumps(get_knowledge_store().rebuild(), ensure_ascii=False, indent=2))


@rag.command("status")
def rag_status():
    """显示索引、图谱和搜索后端的健康状态。"""
    click.echo(json.dumps(get_knowledge_store().status(), ensure_ascii=False, indent=2))


@rag.command("graph")
@click.option("--entity", default="", help="实体 ID；省略时仅显示图谱规模")
def rag_graph(entity):
    """查看由元数据建立的图谱边及其来源分块。"""
    store = get_knowledge_store()
    store.ensure_ready()
    if not entity:
        click.echo(json.dumps(store.status()["graph"], ensure_ascii=False, indent=2))
        return
    with store._connect() as connection:
        edges = [dict(row) for row in connection.execute(
            "SELECT * FROM rag_graph_edges WHERE source=? ORDER BY target, chunk_id LIMIT 200",
            ("entity:" + entity,),
        )]
    click.echo(json.dumps(edges, ensure_ascii=False, indent=2))


@rag.command("search")
@click.argument("query")
@click.option("--top-k", default=5, show_default=True, type=int)
@click.option("--modality", default="")
@click.option("--operator", default="")
@click.option("--document-type", default="")
@click.option("--version", default="")
def rag_search(query, top_k, modality, operator, document_type, version):
    """从命令行执行一次带可选元数据过滤的知识检索。"""
    filters = {key: value for key, value in {
        "document_type": document_type, "version": version,
    }.items() if value}
    results = get_knowledge_store().search(
        query, top_k=top_k, modality=modality, operator=operator,
        metadata_filters=filters,
    )
    click.echo(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))


@rag.command("eval")
@click.option("--file", "dataset_file", type=click.Path(exists=True, path_type=Path))
@click.option("--top-k", default=5, show_default=True, type=int)
@click.option("--details/--no-details", default=False)
@click.option("--ablation/--no-ablation", default=False, help="同时运行查询扩展、精排和父文档消融")
def rag_eval(dataset_file, top_k, details, ablation):
    """运行检索 Golden Dataset，可选执行组件消融对比。"""
    path = dataset_file or PROJECT_ROOT / "rag_knowledge_base" / "evals" / "golden_queries.jsonl"
    if ablation:
        configurations = {
            "baseline_child": {"use_query_expansion": False, "use_decomposition": False, "use_reranker": False, "use_parent": False},
            "reranker_child": {"use_query_expansion": False, "use_decomposition": False, "use_reranker": True, "use_parent": False},
            "reranker_parent": {"use_query_expansion": False, "use_decomposition": False, "use_reranker": True, "use_parent": True},
            "full_rag_fusion": {"use_query_expansion": True, "use_decomposition": True, "use_reranker": True, "use_parent": True},
        }
        reports = {}
        for name, options in configurations.items():
            current = evaluate_retrieval(get_knowledge_store(), path, top_k=top_k, search_options=options)
            current.pop("details", None)
            reports[name] = current
        click.echo(json.dumps(reports, ensure_ascii=False, indent=2))
        return
    report = evaluate_retrieval(get_knowledge_store(), path, top_k=top_k)
    if not details:
        report.pop("details", None)
    click.echo(json.dumps(report, ensure_ascii=False, indent=2))


@rag.command("eval-generation")
@click.option("--file", "dataset_file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--top-k", default=5, show_default=True, type=int)
@click.option("--details/--no-details", default=False)
def rag_eval_generation(dataset_file, top_k, details):
    """评测检索、基于证据的生成、引用和拒答行为。"""
    report = evaluate_generation(get_knowledge_store(), dataset_file, top_k=top_k)
    output = dataset_file.with_name(f"{dataset_file.stem}_generation_result.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not details:
        report.pop("details", None)
    click.echo(json.dumps({"result_file": str(output), **report}, ensure_ascii=False, indent=2))


@cli.group("agent-eval")
def agent_eval():
    """管理并运行受版本控制的 Agent Golden Dataset。"""


@agent_eval.command("seed")
@click.option("--file", "dataset_file", type=click.Path(exists=True, path_type=Path))
def seed(dataset_file):
    """把 JSON 中的标准评测用例同步进数据库。"""
    path = dataset_file or PROJECT_ROOT / "evals" / "golden_dataset.json"
    with session_scope():
        count = seed_golden_cases(path)
    click.echo(f"Seeded {count} golden cases from {path}")


@agent_eval.command("run")
@click.option("--user-id", required=True, type=int)
def run(user_id):
    """为指定用户执行一轮 Agent 回归评测。"""
    with session_scope():
        evaluation = run_golden_evaluation(user_id)
    click.echo(json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
