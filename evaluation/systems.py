"""Retrieval arms, run protocol, and the provenance record written into every report."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import subprocess

from main.retrieval import rag as rag_module
from main.retrieval.rag import MemoryRAG

ARMS = {
    'vector': {'channels': ('semantic',), 'rerank': False},
    'vector_reranker': {'channels': ('semantic',), 'rerank': True},
    'vector_graph': {'channels': ('semantic', 'graph'), 'rerank': False},
    'vector_graph_reranker': {'channels': ('semantic', 'graph'), 'rerank': True},
    # The shipped product path: keyword + semantic + graph, fused and reranked.
    'hybrid_full': {'channels': ('keyword', 'semantic', 'graph'), 'rerank': True},
}
COMPARISONS = (
    ('graph_contribution', 'vector', 'vector_graph'),
    ('reranker_on_vector', 'vector', 'vector_reranker'),
    ('reranker_on_graph', 'vector_graph', 'vector_graph_reranker'),
    ('keyword_contribution', 'vector_graph_reranker', 'hybrid_full'),
)


@dataclass(frozen=True)
class Protocol:
    candidate_limit: int | None = 20   # None measures the shipped default pool formula
    top_k: int = 5                     # sources handed to the answer generator
    ann: bool = False                  # exact search unless the ANN arm is requested
    ann_exact_threshold: int = 1       # with ann=True, force LSH even on small benchmark corpora
    abstain_threshold: float | None = None
    budget: int = 10000
    load_mode: str = 'save'            # 'save' keeps every gold memory; 'ingest' reconciles
    structure: str = 'default'         # post-rerank reordering steps: 'default', 'none', or e.g. 'graph,validity,chain,versions'


def resolve_pool(limit: int, candidate_limit: int | None, *, rerank: bool = True) -> int:
    """The pool a search uses; None resolves the library default for a reranked or fusion-only arm."""
    if candidate_limit is not None:
        return candidate_limit
    return rag_module.default_pool(limit, rerank=rerank)


def build_rag(store, *, embedder, protocol: Protocol, index_path=None, reranker='auto') -> MemoryRAG:
    options = {'ann_enabled': protocol.ann}
    if protocol.ann:
        options['ann_exact_threshold'] = protocol.ann_exact_threshold
    rag = MemoryRAG(store, embedder=embedder, index_path=index_path, reranker=reranker, **options)
    if protocol.structure != 'default':
        steps = frozenset() if protocol.structure == 'none' else frozenset(protocol.structure.split(','))
        if steps - rag_module.ALL_STRUCTURE_STEPS:
            raise ValueError(f'Unknown structure steps: {sorted(steps - rag_module.ALL_STRUCTURE_STEPS)}')
        rag.structure_steps = steps
    return rag


def search_options(arm: str, *, as_of=None) -> dict:
    spec = ARMS[arm]
    return {'mode': 'hybrid', 'channels': spec['channels'], 'rerank': spec['rerank'], 'as_of': as_of}


def answer_extractive(context: dict, threshold: float | None) -> dict:
    from main.retrieval.generation import answer
    if threshold is None:
        return answer(context, generator='extractive')
    return answer(context, generator='extractive', abstain_threshold=threshold)


def git_head() -> str:
    try:
        head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout.strip()
        return head + ('+dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def protocol_record(protocol: Protocol, *, embedder=None, rag=None, **extra) -> dict:
    reranker = getattr(rag, '_reranker', None) if rag is not None else None
    record = {**asdict(protocol),
              'embedding_model': getattr(embedder, 'model_id', None),
              'reranker_model': getattr(reranker, 'model_id', None),
              'reranker_threads': getattr(reranker, 'threads', None),
              'ann_exact_threshold': getattr(rag, 'ann_exact_threshold', None),
              'default_pool_formula': 'reranked: min(200, max(100, 4*limit)); fusion only: min(200, max(20, 4*limit))',
              'git_head': git_head(), 'python': platform.python_version()}
    record.update(extra)
    return record


class CachingEmbedder:
    """Wrap an embedder so repeated queries across arms are encoded once."""

    def __init__(self, inner, queries=()):
        self.inner, self.model_id, self.dimension = inner, inner.model_id, inner.dimension
        self.cache = {}
        unique = list(dict.fromkeys(queries))
        if unique:
            self.cache.update(zip(unique, inner.queries(unique)))

    def passages(self, texts):
        return self.inner.passages(texts)

    def queries(self, texts):
        missing = [text for text in dict.fromkeys(texts) if text not in self.cache]
        if missing:
            self.cache.update(zip(missing, self.inner.queries(missing)))
        return [self.cache[text] for text in texts]

    def query(self, text):
        return self.queries([text])[0]
