"""Test package for unittest discovery."""
import os

# Tests choose their importance scorer explicitly (MemoryCore.with_heuristic(), --scorer, Settings.for_testing);
# a developer's COGNIMEM_SCORER must not change what they measure, and an exported COGNIMEM_JWT_SECRET must not make
# `import api.app` build the real hybrid scorer at import time.
os.environ.pop("COGNIMEM_SCORER", None)
os.environ.pop("COGNIMEM_JWT_SECRET", None)
