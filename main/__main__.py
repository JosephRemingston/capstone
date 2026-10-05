"""Command-line interface for the local Memory Core."""

from __future__ import annotations

import argparse
import json
import os
import sys
import sqlite3
from pathlib import Path
from typing import Any

from .application.core import MemoryCore
from .domain.models import MemoryCategory, MemoryInput, MemoryRecord, MemoryTier
from .storage.store import DEFAULT_STORE_PATH, LocalMemoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m main",
        description="Process and inspect CogniMem memory records using a local JSONL store.",
    )
    parser.add_argument(
        "--store",
        default=os.environ.get("COGNIMEM_STORE", str(DEFAULT_STORE_PATH)),
        help="Path to the local JSONL memory store. Defaults to memory_store/memories.jsonl.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process content into a MemoryRecord.")
    process.add_argument("content", help="Raw message content to process.")
    process.add_argument("--user-id", required=True, help="User identifier.")
    process.add_argument("--session-id", required=True, help="Session identifier.")
    process.add_argument("--role", default="user", help="Message role. Defaults to user.")
    process.add_argument("--metadata", default="{}", help="JSON object containing optional metadata.")
    process.add_argument("--interaction-score", type=float, help="Shortcut for metadata.interaction_score.")
    process.add_argument("--no-save", action="store_true", help="Process and print the record without saving it.")
    process.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    process.add_argument("--scorer", choices=["heuristic", "xgboost", "neural"], default="heuristic",
                         help="Importance scorer; trained options use experimental retention proxy models.")
    process.add_argument("--model-dir", type=Path, help="Custom model artifact directory (requires --scorer xgboost or neural).")
    process.add_argument("--split", action="store_true", help="Split independent clauses; return an array of memories.")
    process.add_argument("--due-at", help="Explicit ISO deadline in the input timezone (UTC by default).")
    process.add_argument("--task-id", help="Explicit task to complete, cancel, or reschedule.")
    process.add_argument("--occurrence-at", help="ISO date/datetime of the current recurring occurrence.")
    process.add_argument("--task-scope", choices=['occurrence', 'series'], help="Update one occurrence (default) or the entire series.")
    process.add_argument('--confidence', type=float, help='Assertion confidence, from 0 to 1.')
    process.add_argument('--source-priority', type=float, help='Caller-assigned source priority, from 0 to 100.')

    list_cmd = subparsers.add_parser("list", help="List stored memory records.")
    add_filter_args(list_cmd)
    list_cmd.add_argument("--limit", type=int, default=20, help="Maximum records to return. Defaults to 20.")
    list_cmd.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    search = subparsers.add_parser("search", help="Rank keyword matches by relevance, category, tier, recency, and importance.")
    search.add_argument("query", help="Search query.")
    add_filter_args(search)
    search.add_argument("--limit", type=int, default=20, help="Maximum records to return. Defaults to 20.")
    search.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    get = subparsers.add_parser("get", help="Get one memory record by ID.")
    get.add_argument("record_id", help="Memory record ID.")
    get.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    stats = subparsers.add_parser("stats", help="Show local store counts by category and tier.")
    stats.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

    history = subparsers.add_parser('history', help='Show every persisted revision of a memory.')
    history.add_argument('record_id')
    history.add_argument('--user-id', required=True)
    history.add_argument('--pretty', action='store_true')

    resolve = subparsers.add_parser('resolve', help='Explicitly select the current assertion in a conflict.')
    resolve.add_argument('--keep', required=True, help='ID of the fact/preference root to keep.')
    resolve.add_argument('--user-id', required=True)
    resolve.add_argument('--pretty', action='store_true')

    reconcile = subparsers.add_parser('reconcile', help='Resolve conflicts and consolidate existing facts/preferences.')
    reconcile.add_argument('--user-id', required=True)
    reconcile.add_argument('--pretty', action='store_true')

    search.add_argument('--mode', choices=['keyword', 'semantic', 'hybrid', 'graph'], default='keyword')
    search.add_argument('--no-rerank', action='store_false', dest='rerank', default=None,
                        help='Disable the local cross-encoder for hybrid search.')
    search.add_argument('--candidate-limit', type=int,
                        help='Fused candidates considered before Top-K reranking (default: max(20, 4×limit)).')
    for name in ('ask', 'index', 'graph'):
        command = subparsers.add_parser(name, help={'ask': 'Answer using retrieved memory evidence.',
                                                   'index': 'Build or update vector and temporal graph indexes.',
                                                   'graph': 'Inspect temporal relationships and evidence paths.'}[name])
        command.add_argument('--user-id', required=True)
        command.add_argument('--pretty', action='store_true')
        if name == 'index':
            command.add_argument('--graph-only', action='store_true')
        else:
            command.add_argument('--as-of', help='Valid time, ISO datetime with timezone.')
            command.add_argument('--known-at', help='Recording-time cutoff, ISO datetime with timezone.')
            command.add_argument('--hops', type=int, default=2)
            if name == 'ask':
                command.add_argument('query')
                command.add_argument('--mode', choices=['keyword', 'semantic', 'hybrid', 'graph'], default='hybrid')
                command.add_argument('--generator', choices=['gemini', 'extractive'], default='gemini')
                command.add_argument('--limit', type=int, default=5)
                command.add_argument('--no-rerank', action='store_false', dest='rerank', default=None,
                                     help='Disable the local cross-encoder for hybrid retrieval.')
                command.add_argument('--candidate-limit', type=int,
                                     help='Fused candidates considered before Top-K reranking.')
                command.add_argument('--budget', type=int, default=10000, help='Evidence context character budget.')
                command.add_argument('--preview', action='store_true', help='Show context and prompt without an API call.')
                command.add_argument('--env-file', default='.env')
            else:
                command.add_argument('--query', help='Entity names to seed traversal; omitted lists relationships.')
                command.add_argument('--predicate')
                command.add_argument('--direction', choices=['both', 'outgoing', 'incoming'], default='both')
    cleanup_cmd = subparsers.add_parser('cleanup', help='Preview or apply safe expiry cleanup and history compaction.')
    cleanup_cmd.add_argument('--user-id', required=True)
    cleanup_cmd.add_argument('--grace-days', type=float, default=7)
    cleanup_cmd.add_argument('--apply', action='store_true', help='Physically delete eligible data; default is a preview.')
    cleanup_cmd.add_argument('--scheduled', action='store_true', help='Run only when the persisted schedule is due.')
    cleanup_cmd.add_argument('--interval-hours', type=float, default=24)
    cleanup_cmd.add_argument('--retention-policy', choices=['balanced', 'expiry'], default='balanced',
                             help='Balanced archives useful/sensitive expired memories; expiry deletes all eligible data.')
    cleanup_cmd.add_argument('--pretty', action='store_true')
    return parser


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--include-history', action='store_true', help='Include superseded and consolidated records.')
    parser.add_argument("--include-expired", action="store_true", help="Include expired records for inspection.")
    parser.add_argument("--include-resolved", action="store_true", help="Include completed/cancelled tasks.")
    parser.add_argument("--user-id", help="Filter by user identifier.")
    parser.add_argument("--session-id", help="Filter by session identifier.")
    parser.add_argument("--category", choices=[category.value for category in MemoryCategory], help="Filter by category.")
    parser.add_argument("--tier", choices=[tier.value for tier in MemoryTier], help="Filter by memory tier.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = LocalMemoryStore(Path(args.store))

    try:
        if args.command == 'cleanup':
            from .storage.cleanup import cleanup
            print_json(cleanup(store, user_id=args.user_id, grace_days=args.grace_days,
                               apply=args.apply, scheduled=args.scheduled, interval_hours=args.interval_hours,
                               retention_policy=args.retention_policy), pretty=args.pretty)
            return 0
        if args.command == "process":
            return process_command(args, store)
        if args.command == "list":
            records = store.list(
                user_id=args.user_id,
                session_id=args.session_id,
                category=args.category,
                tier=args.tier,
                limit=args.limit,
                include_expired=args.include_expired,
                include_resolved=args.include_resolved,
                include_history=args.include_history,
            )
            print_json([record.to_dict() for record in records], pretty=args.pretty)
            return 0
        if args.command in {'ask', 'index', 'graph'} or (args.command == 'search' and args.mode != 'keyword'):
            return rag_command(args, store)
        if args.command == "search":
            records = store.search(
                args.query,
                user_id=args.user_id,
                session_id=args.session_id,
                category=args.category,
                tier=args.tier,
                limit=args.limit,
                include_expired=args.include_expired,
                include_resolved=args.include_resolved,
                include_history=args.include_history,
            )
            print_json([record.to_dict() for record in records], pretty=args.pretty)
            return 0
        if args.command == "get":
            record = store.get(args.record_id)
            if record is None:
                print_json({"error": "record_not_found", "record_id": args.record_id}, pretty=args.pretty)
                return 1
            print_json(record.to_dict(), pretty=args.pretty)
            return 0
        if args.command == "stats":
            print_json(build_stats(store.all(), store.path), pretty=args.pretty)
            return 0
        if args.command == 'history':
            records = store.history(args.record_id, user_id=args.user_id)
            print_json([record.to_dict() for record in records], pretty=args.pretty)
            return 0
        if args.command == 'resolve':
            record = store.resolve_conflict(args.keep, user_id=args.user_id)
            print_json(record.to_dict(), pretty=args.pretty)
            return 0
        if args.command == 'reconcile':
            records = store.reconcile_memories(user_id=args.user_id)
            print_json({'changed_records': len(records), 'records': [record.to_dict() for record in records]}, pretty=args.pretty)
            return 0
    except (OSError, sqlite3.Error) as exc:
        if args.command != 'cleanup':
            raise
        print_json({'error': 'cleanup_failed', 'error_type': type(exc).__name__,
                    'message': 'Cleanup failed; retry after resolving the storage/index error.'}, pretty=args.pretty)
        return 1
    except ValueError as exc:
        print_json({"error": "invalid_input", "message": str(exc)}, pretty=getattr(args, "pretty", False))
        return 2

    parser.print_help()
    return 2


def rag_command(args, store):
    from .retrieval.rag import MemoryRAG
    from .graph import parse_time
    from .retrieval.generation import answer, prompt
    rag = MemoryRAG(store)
    if args.command == 'index':
        payload = rag.sync(user_id=args.user_id, semantic=not args.graph_only)
    elif args.command == 'search':
        payload = rag.search(args.query, user_id=args.user_id, mode=args.mode, limit=args.limit,
                             rerank=args.rerank, candidate_limit=args.candidate_limit,
                             session_id=args.session_id, category=args.category, tier=args.tier,
                             include_expired=args.include_expired, include_resolved=args.include_resolved,
                             include_history=args.include_history)
    else:
        as_of, known_at = parse_time(args.as_of), parse_time(args.known_at)
        if args.command == 'graph':
            rag.sync(user_id=args.user_id, semantic=False)
            if args.query:
                entities = rag.graph.entities(args.query, user_id=args.user_id, known_at=known_at)
                paths = rag.graph.traverse([e['id'] for e in entities], user_id=args.user_id,
                                          as_of=as_of, known_at=known_at, max_hops=args.hops, direction=args.direction)
                if args.predicate:
                    paths = [p for p in paths if any(e['predicate'] == args.predicate for e in p['edges'])]
                payload = {'entities': entities, 'paths': paths}
            else:
                payload = rag.graph.relations(user_id=args.user_id, as_of=as_of, known_at=known_at, predicate=args.predicate)
        else:
            context = rag.context(args.query, user_id=args.user_id, mode=args.mode, limit=args.limit,
                                  budget=args.budget, as_of=as_of, known_at=known_at, max_hops=args.hops,
                                  rerank=args.rerank, candidate_limit=args.candidate_limit)
            payload = {'context': context, 'prompt': prompt(context)} if args.preview else answer(
                context, generator=args.generator, env_file=args.env_file)
    print_json(payload, pretty=args.pretty)
    return 0


def process_command(args: argparse.Namespace, store: LocalMemoryStore) -> int:
    metadata = parse_metadata(args.metadata)
    if args.interaction_score is not None:
        metadata["interaction_score"] = args.interaction_score
    if args.due_at is not None:
        metadata["due_at"] = args.due_at
    if args.task_id is not None:
        metadata["task_id"] = args.task_id
    if args.occurrence_at is not None:
        metadata['occurrence_at'] = args.occurrence_at
    if args.task_scope is not None:
        metadata['task_scope'] = args.task_scope
    if args.confidence is not None:
        metadata['confidence'] = args.confidence
    if args.source_priority is not None:
        metadata['source_priority'] = args.source_priority

    memory_input = MemoryInput(
        content=args.content,
        user_id=args.user_id,
        session_id=args.session_id,
        role=args.role,
        metadata=metadata,
    )
    if args.model_dir is not None and args.scorer == "heuristic":
        raise ValueError("--model-dir requires --scorer xgboost or neural")
    core = (MemoryCore.with_ml(args.model_dir) if args.scorer == "xgboost"
            else MemoryCore.with_neural(args.model_dir) if args.scorer == "neural"
            else MemoryCore())
    records = core.process_many(memory_input) if args.split else [core.process(memory_input)]
    if not args.no_save:
        records = [store.ingest(record) for record in records]
    payload = [record.to_dict() for record in records]
    print_json(payload if args.split else payload[0], pretty=args.pretty)
    return 0


def parse_metadata(raw_metadata: str) -> dict[str, Any]:
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise ValueError("--metadata must be a valid JSON object") from exc
    if not isinstance(metadata, dict):
        raise ValueError("--metadata must be a JSON object")
    return metadata


def build_stats(records: list[MemoryRecord], path: Path) -> dict[str, Any]:
    by_category = {category.value: 0 for category in MemoryCategory}
    by_tier = {tier.value: 0 for tier in MemoryTier}
    for record in records:
        by_category[record.category.value] += 1
        by_tier[record.tier.value] += 1

    return {
        "store": str(path),
        "total_records": len(records),
        "by_category": by_category,
        "by_tier": by_tier,
    }


def print_json(payload: Any, *, pretty: bool = False) -> None:
    if pretty:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())

# PR