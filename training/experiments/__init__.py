"""Isolated retention experiment: XGBoost and frozen-BGE heads on three datasets.

Nothing here touches ``main/``, the shipped artifacts under ``artifacts/importance*`` or the
tracked retention reports. Working files go to the git-ignored ``artifacts/experiments/`` and
``data/experiments/``; the only tracked output is ``evaluation/reports/experiments/``.

Process isolation: ``xgboost_run`` imports xgboost and never torch, ``neural_run`` imports torch
and never xgboost; ``datasets``, ``features``, ``metrics``, ``combine`` and ``report`` import
neither native library and are safe to import from either process.
"""
