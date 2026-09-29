"""Audit memory records and gold references for measurable data-quality properties."""
from __future__ import annotations

from datetime import datetime
import statistics


def audit(payload):
    memories = payload['memories']
    identifiers = [row['id'] for row in memories]
    required = ('id', 'content', 'user_id', 'session_id', 'category', 'tier',
                'created_at', 'updated_at', 'recorded_at', 'confidence', 'source_metadata')
    complete = [sum(row.get(key) is not None for key in required) / len(required) for row in memories]
    lags = [(datetime.fromisoformat(row['recorded_at']) - datetime.fromisoformat(row['created_at'])).total_seconds()
            for row in memories]
    references = [identifier for query in payload['queries']
                  for identifier in [*query['relevant_memory_ids'], *query['supporting_memory_ids']]]
    metadata = [row['source_metadata'] for row in memories]
    graph_relations = [relation for item in metadata for relation in item.get('relations', [])]
    return {
        'accuracy': {'status': 'requires_independent_review',
                     'reason': 'Template construction cannot independently verify factual truth.'},
        'completeness': {'mean_required_field_coverage': statistics.mean(complete),
                         'records_with_all_required_fields': sum(value == 1 for value in complete),
                         'records': len(memories)},
        'consistency': {'unique_memory_ids': len(set(identifiers)) == len(identifiers),
                        'all_query_references_exist': set(references) <= set(identifiers)},
        'timeliness': {'mean_recording_lag_seconds': statistics.mean(lags),
                       'max_recording_lag_seconds': max(lags)},
        'provenance': {'speaker_coverage': sum(bool(item.get('speaker')) for item in metadata) / len(metadata),
                       'confidence_coverage': sum('confidence' in row for row in memories) / len(memories),
                       'explicit_temporal_relations': sum('valid_from' in relation for relation in graph_relations),
                       'explicit_relationships': len(graph_relations)},
        'conflict_state': {'conflict_queries': sum(row['query_type'] == 'conflict' for row in payload['queries']),
                           'graded_current_and_historical_evidence': all(
                               len(row['graded_relevance']) >= 2 for row in payload['queries']
                               if row['query_type'] == 'conflict')},
    }
