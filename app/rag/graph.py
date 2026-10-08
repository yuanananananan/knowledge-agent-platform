"""文件用途：构建带来源证据的实体、文档和声明图谱，并支持有界遍历。

Edges describe documented associations, never inferred causality or measured benefit.
Traversal is bounded to entity -> document and only returns caller-visible chunks.
"""
from __future__ import annotations
import json
import sqlite3
from collections import defaultdict


def schema(connection: sqlite3.Connection) -> None:
    """创建知识图谱节点表、边表和查询索引。"""
    connection.executescript('''
        CREATE TABLE IF NOT EXISTS rag_graph_nodes (
            node_id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS rag_graph_edges (
            source TEXT NOT NULL, relation TEXT NOT NULL, target TEXT NOT NULL,
            chunk_id TEXT NOT NULL, document_path TEXT NOT NULL,
            PRIMARY KEY(source, relation, target, chunk_id)
        );
        CREATE INDEX IF NOT EXISTS ix_graph_source ON rag_graph_edges(source, relation);
    ''')


def rebuild(connection: sqlite3.Connection) -> None:
    """根据当前知识分块的元数据重建实体—文档—声明关系图。"""
    nodes, edges = {}, set()
    for row in connection.execute('SELECT chunk_id, document_path, metadata FROM rag_chunks'):
        chunk, path, raw = row
        metadata = json.loads(raw)
        document = 'document:' + path
        nodes[document] = ('document', path)
        kind = metadata.get('document_type', '')
        relation = ('participates_in' if kind in {'planned_experiment', 'experiment_plan', 'validation_case', 'selection_case'}
                    else 'documented_in')
        for entity in metadata.get('entity_ids', []):
            node = 'entity:' + entity
            nodes[node] = ('entity', entity)
            edges.add((node, relation, document, chunk, path))
        for claim in metadata.get('claim_ids', []):
            node = 'claim:' + claim
            nodes[node] = ('claim', claim)
            edges.add((document, 'supports_claim', node, chunk, path))
        for key in ('modality', 'version'):
            if metadata.get(key):
                value = str(metadata[key])
                node = key + ':' + (path + ':' if key == 'version' else '') + value
                nodes[node] = (key, value)
                edges.add((document, 'has_' + key, node, chunk, path))
    connection.execute('DELETE FROM rag_graph_edges')
    connection.execute('DELETE FROM rag_graph_nodes')
    connection.executemany('INSERT INTO rag_graph_nodes VALUES (?, ?, ?)',
                           [(node, *value) for node, value in sorted(nodes.items())])
    connection.executemany('INSERT INTO rag_graph_edges VALUES (?, ?, ?, ?, ?)', sorted(edges))
    connection.execute("INSERT OR REPLACE INTO rag_meta VALUES ('graph_version', '1')")


def candidates(connection, rows, seed_entities: list[str], fused: list[float], limit: int = 6):
    """为每个关联文档选出最佳分块，并附上明确的来源路径。

    图谱扩展只允许使用调用方已筛选的分块，不能绕过元数据、访问范围或旧版本过滤。
    """
    if not seed_entities:
        return {}
    eligible = {row['chunk_id']: i for i, (row, _) in enumerate(rows)}
    matches = defaultdict(list)
    for entity in dict.fromkeys(seed_entities):
        for edge in connection.execute(
            'SELECT target, chunk_id, document_path, relation FROM rag_graph_edges WHERE source=?',
            ('entity:' + entity,),
        ):
            target, chunk, path, relation = edge
            if chunk in eligible:
                matches[path].append((entity, eligible[chunk], relation))
    ranked = []
    for path, values in matches.items():
        index = max({v[1] for v in values}, key=lambda i: fused[i])
        entities = sorted({v[0] for v in values})
        # Shared cases are useful bridges; single-entity documentation remains
        # relevant but must compete on original retrieval score.
        bridge = len(entities) >= 2 and any(v[2] == 'participates_in' for v in values)
        ranked.append((bridge, len(entities), fused[index], index, [
            {'entity': e, 'relation': relation, 'document': path, 'chunk_id': rows[i][0]['chunk_id']}
            for e, i, relation in values if i == index
        ]))
    ranked.sort(key=lambda x: x[:3], reverse=True)
    return {index: {'bridge': bridge, 'paths': paths} for bridge, _, _, index, paths in ranked[:limit]}
