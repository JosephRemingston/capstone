"""Remove every generated, downloaded, cached, or trained file so CogniMem can be rebuilt from source.

Dry run by default. Pass --apply to delete. Only the allow-listed paths below are touched;
user documents under data/ and hand-written notes under docs/ are protected explicitly.

    python scripts/clean_slate.py            # list what would be removed
    python scripts/clean_slate.py --apply    # remove it
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

# Directories or files removed outright (relative to ROOT).
TARGETS = (
    'artifacts/ensemble_training',
    'frontend/node_modules',
    'frontend/dist',
    'data/rag/models',
    'data/evaluation/locomo',
    'data/evaluation/longmemeval',
    'data/evaluation/babi',
    'data/evaluation/personalization',
    'data/evaluation/retrieval_training',
    'data/halumem',
    'data/hippocorpus',
    'docs/reports',
    'memory_store',
    'tests/evaluation',
)
# Name patterns removed anywhere in the repository, outside .git and .venv.
PATTERNS = ('__pycache__', '.DS_Store', '.pytest_cache')
SKIP_DIRS = {'.git', '.venv'}
# Never removed, even if a target or pattern would match them.
PROTECTED = (
    'data/Capstone Project Proposal.pdf',
    'data/Capstone_Base_Paper_Analysis.docx',
    'data/RAG-Driven_Memory_Architectures_in_Conversational_LLMsA_Literature_Review_With_Insights_Into_Emerging_Agriculture_Data_Sharing.pdf',
    'data/docs',
    'docs/research',
    '.env',
    'reviews',
    'static',
    'PROJECT_VISION.md',
    'artifacts/importance_ensemble',   # the tracked default importance model
)


def size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    return sum(item.lstat().st_size for item in path.rglob('*') if item.is_file() or item.is_symlink())


def protected(path: Path) -> bool:
    resolved = path.resolve()
    for item in PROTECTED:
        guard = (ROOT / item).resolve()
        if resolved == guard or guard in resolved.parents or resolved in guard.parents:
            return True
    return False


def pattern_matches(root: Path):
    for path in sorted(root.iterdir()):
        if path.name in SKIP_DIRS:
            continue
        if path.name in PATTERNS:
            yield path
        elif path.is_dir() and not path.is_symlink():
            yield from pattern_matches(path)


def plan() -> list[Path]:
    found = [ROOT / item for item in TARGETS if (ROOT / item).exists() or (ROOT / item).is_symlink()]
    found += [path for path in pattern_matches(ROOT) if not any(parent in found for parent in path.parents)]
    for path in found:
        if protected(path):
            raise SystemExit(f'Refusing to remove protected path: {path.relative_to(ROOT)}')
    return found


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true', help='delete the listed paths (default: dry run)')
    args = parser.parse_args()
    targets = plan()
    total = 0
    for path in targets:
        bytes_ = size(path)
        total += bytes_
        print(f"{'removed' if args.apply else 'would remove'}  {bytes_ / 1e6:9.2f} MB  {path.relative_to(ROOT)}")
        if args.apply:
            remove(path)
    print(f"{'Freed' if args.apply else 'Would free'} {total / 1e6:.2f} MB across {len(targets)} paths.")
    if not args.apply:
        print('Dry run only. Re-run with --apply to delete.')


if __name__ == '__main__':
    main()
