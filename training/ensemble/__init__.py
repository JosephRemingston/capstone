"""Training for the default importance model: the csv5k "Ensemble + thresholds".

    python -m training.ensemble.run

Trains a fit-only XGBoost model and three seeds of the frozen-BGE deep head on the csv5k importance
dataset, tunes temperatures, the blend weight and class weights on validation, scores the test split once,
and writes the serving artifact ``artifacts/importance_ensemble/`` plus the tracked report
``evaluation/reports/retention_ensemble.{json,md}``. Working files go to the git-ignored
``artifacts/ensemble_training/``.

Process isolation: ``xgboost_stage`` imports xgboost and never torch, ``neural_stage`` imports torch and never
xgboost; ``assemble`` loads the serving model (xgboost + onnxruntime) without torch. ``datasets``,
``features``, ``metrics`` and ``report`` import no native ML library.
"""
