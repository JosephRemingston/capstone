"""Pin every wall-clock read of the memory pipeline to one instant, so runner results do not depend on the day."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import datetime
from unittest.mock import patch

# The three places the pipeline reads the clock: the heuristic's recency feature, lifecycle archiving, store visibility.
TARGETS = ('main.domain.features.FeatureExtractor._now', 'main.domain.lifecycle.utc_now', 'main.storage.store.utc_now')


@contextmanager
def pinned_clock(when: datetime):
    with ExitStack() as stack:
        for target in TARGETS:
            stack.enter_context(patch(target, return_value=when))
        yield
