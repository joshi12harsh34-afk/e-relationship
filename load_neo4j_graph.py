#!/usr/bin/env python3
"""Load one extraction JSON confirmed graph into Neo4j (per-file scoped).

Each file gets its own subgraph keyed by ``graph_key`` (JSON stem). Re-running
the same file replaces only that file's nodes/edges.

Env (or flags):
  NEO4J_URI      default bolt://localhost:7687
  NEO4J_USER     default neo4j
  NEO4J_PASSWORD required

Example:
  python load_neo4j_graph.py --file "outputs/Annexure 2.3 _ 1916-PRO-TDS-22-PK-3901_advanced_auto_extraction.json"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_dotenv(path: Path = ROOT / '.env') -> None:
    """Minimal .env loader (no extra dependency).

    Fills missing or blank env vars from ``.env``. Non-empty existing values win.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, val = line.partition('=')
        key = key.strip()
        if not key:
            continue
        existing = os.environ.get(key)
        if existing is not None and str(existing).strip():
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in '"\'':
            val = val[1:-1]
        os.environ[key] = val


_load_dotenv()

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    print('Install driver: pip install neo4j', file=sys.stderr)
    raise SystemExit(1)


_NON_ALNUM = re.compile(r'[^A-Za-z0-9]+')


def _neo_label(raw: str, fallback: str = 'Entity') -> str:
    text = _NON_ALNUM.sub('', str(raw or '').strip())
    if not text:
        return fallback
    if text[0].isdigit():
        text = f'L{text}'
    return text[:64]


def _neo_rel_type(raw: str) -> str:
    parts = [p for p in _NON_ALNUM.split(str(raw or '').strip().upper()) if p]
    name = '_'.join(parts) or 'RELATED_TO'
    if name[0].isdigit():
        name = f'R_{name}'
    return name[:64]


def _graph_key(path: Path, payload: dict) -> str:
    source = str((payload.get('graph') or {}).get('source_file') or path.stem)
    stem = Path(source).stem or path.stem
    return stem[:120]


def load_payload(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('graph'), dict):
        raise ValueError(f'No graph object in {path}')
    return data


def replace_file_graph(session, *, graph_key: str, source_file: str, nodes: list, edges: list) -> tuple[int, int]:
    """Delete prior subgraph for this file, then write nodes + edges."""
    session.run(
        'MATCH (n:ExtractedEntity {graph_key: $graph_key}) DETACH DELETE n',
        graph_key=graph_key,
    )

    node_count = 0
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        nid = str(node.get('id') or '').strip()
        if not nid:
            continue
        primary = _neo_label(node.get('label') or 'Entity')
        props = {
            'id': nid,
            'graph_key': graph_key,
            'source_file': source_file,
            'text': str(node.get('text') or ''),
            'label': str(node.get('label') or ''),
            'subtype': str(node.get('subtype') or ''),
        }
        # Dynamic primary label + stable ExtractedEntity for cleanup/query.
        cypher = (
            f'MERGE (n:ExtractedEntity:`{primary}` {{graph_key: $graph_key, id: $id}}) '
            'SET n += $props'
        )
        session.run(cypher, graph_key=graph_key, id=nid, props=props)
        node_count += 1

    edge_count = 0
    for edge in edges or []:
        if not isinstance(edge, dict):
            continue
        sid = str(edge.get('subject_id') or '').strip()
        if not sid:
            continue
        pred = _neo_rel_type(edge.get('predicate') or 'RELATED_TO')
        oid = str(edge.get('object_id') or '').strip()
        literal = edge.get('object_literal')
        if literal is None and edge.get('object') is not None and not oid:
            literal = edge.get('object')

        common = {
            'graph_key': graph_key,
            'source_file': source_file,
            'predicate': str(edge.get('predicate') or ''),
            'kind': str(edge.get('kind') or ''),
            'relation_category': str(edge.get('relation_category') or ''),
            'evidence_status': str(edge.get('evidence_status') or ''),
        }

        if oid:
            cypher = (
                'MATCH (s:ExtractedEntity {graph_key: $graph_key, id: $sid}) '
                'MATCH (o:ExtractedEntity {graph_key: $graph_key, id: $oid}) '
                f'MERGE (s)-[r:`{pred}` {{graph_key: $graph_key, sid: $sid, oid: $oid}}]->(o) '
                'SET r += $props'
            )
            session.run(
                cypher,
                graph_key=graph_key,
                sid=sid,
                oid=oid,
                props={**common, 'oid': oid, 'sid': sid},
            )
            edge_count += 1
            continue

        if literal is None:
            continue
        lit_id = f'lit:{sid}:{pred}:{hash(str(literal)) & 0xFFFFFFFF:x}'
        cypher = (
            'MATCH (s:ExtractedEntity {graph_key: $graph_key, id: $sid}) '
            'MERGE (o:ExtractedEntity:Literal {graph_key: $graph_key, id: $lit_id}) '
            'SET o.text = $literal, o.label = "Literal", o.source_file = $source_file '
            f'MERGE (s)-[r:`{pred}` {{graph_key: $graph_key, sid: $sid, oid: $lit_id}}]->(o) '
            'SET r += $props'
        )
        session.run(
            cypher,
            graph_key=graph_key,
            sid=sid,
            lit_id=lit_id,
            literal=str(literal),
            source_file=source_file,
            props={**common, 'sid': sid, 'oid': lit_id, 'object_literal': str(literal)},
        )
        edge_count += 1

    return node_count, edge_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Load one per-file confirmed graph into Neo4j')
    parser.add_argument(
        '--file',
        type=Path,
        required=True,
        help='Path to *_extraction.json',
    )
    parser.add_argument('--uri', default=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'))
    parser.add_argument('--user', default=os.environ.get('NEO4J_USER', 'neo4j'))
    parser.add_argument('--password', default=os.environ.get('NEO4J_PASSWORD', ''))
    args = parser.parse_args(argv)

    path = args.file if args.file.is_absolute() else ROOT / args.file
    if not path.is_file():
        print(f'File not found: {path}', file=sys.stderr)
        return 1
    if not args.password:
        print('Set NEO4J_PASSWORD (or --password).', file=sys.stderr)
        return 1

    payload = load_payload(path)
    graph = payload.get('graph') or {}
    source_file = str(graph.get('source_file') or path.name)
    key = _graph_key(path, payload)
    nodes = list(graph.get('nodes') or [])
    edges = list(graph.get('edges') or [])

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as session:
            n, e = replace_file_graph(
                session,
                graph_key=key,
                source_file=source_file,
                nodes=nodes,
                edges=edges,
            )
    except Exception as exc:
        name = type(exc).__name__
        if 'AuthError' in name or 'Unauthorized' in str(exc):
            print(
                'Neo4j auth failed. Check NEO4J_PASSWORD in .env '
                '(or pass --password). If the shell already has a wrong '
                'NEO4J_PASSWORD, clear it: Remove-Item Env:NEO4J_PASSWORD',
                file=sys.stderr,
            )
            return 1
        raise
    finally:
        driver.close()

    print(f'Loaded per-file graph_key={key!r}: nodes={n}, edges={e}')
    print('Neo4j Browser:')
    print(f"  MATCH (n:ExtractedEntity {{graph_key: '{key}'}})-[r]-(m) RETURN n,r,m")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
