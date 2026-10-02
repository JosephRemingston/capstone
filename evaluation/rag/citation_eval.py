"""Mechanical citation integrity and gold-support coverage metrics."""
from __future__ import annotations


def evaluate_citations(answer, context, supporting_ids):
    citations = [item for statement in answer.get('statements', [])
                 for item in statement.get('evidence', [])]
    cited_ids = {item.get('source_id') for item in citations}
    existing = {key for key in cited_ids if key in context['sources']}
    quote_matches = sum(item.get('source_id') in context['sources'] and
                        isinstance(item.get('quote'), str) and item['quote'] in
                        context['sources'][item['source_id']]['text'] for item in citations)
    supporting = set(supporting_ids)
    supported_citations = sum(item.get('source_id') in supporting for item in citations)
    return {
        'citation_id_validity': len(existing) / len(cited_ids) if cited_ids else float(answer.get('abstain', False)),
        'citation_quote_validity': quote_matches / len(citations) if citations else float(answer.get('abstain', False)),
        'citation_precision': supported_citations / len(citations) if citations else float(answer.get('abstain', False)),
        'citation_recall': len(cited_ids & supporting) / len(supporting) if supporting else float(answer.get('abstain', False)),
        'citation_completeness': float(supporting <= cited_ids) if supporting else float(answer.get('abstain', False)),
    }
