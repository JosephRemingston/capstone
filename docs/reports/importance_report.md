# Hippocorpus importance-model experiment

## Result

XGBoost was trained on existing human-reported personal-event importance ratings.
It is an experimental proxy for conversational memory importance, not a validated
replacement for the retention policy. No accuracy percentage is reported because
this is regression on an ordinal rating mapped to 0–1.

| Model | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ | Spearman ↑ |
|---|---:|---:|---:|---:|
| xgboost | 0.2291 | 0.2847 | 0.0311 | 0.2367 |
| training_mean | 0.2424 | 0.2947 | -0.0381 | undefined (constant) |
| existing_heuristic | 0.3098 | 0.3612 | -0.5593 | 0.0470 |

The model improves test RMSE by about 3.4% over the training-mean baseline, but
explains only about 3.1% of test variance. The heuristic comparison is contextual:
it was designed for conversational retention, not this dataset's rating target.

## Data and split

- Source: [official Microsoft Hippocorpus release](https://www.microsoft.com/en-us/download/details.aspx?id=105291).
- Release archive: `hippocorpus-u20220112.zip`, containing `hippoCorpusV2.csv`.
- 6,854 total stories; 6,710 used; 144 missing importance ratings excluded.
- Rating values verified as 1–5; target is `(importance - 1) / 4`.
- Train: 4,697; validation: 1,006; test: 1,007.
- Connected components link matching authors, recalled/imagined/retold pair IDs,
  and duplicate story/summary text. Components never cross splits. Unlabeled rows
  also participate in component construction before label filtering.
- The largest connected component is reserved for training; other components are
  shuffled with seed 42, targeting 15% each for test and validation.
- Train/validation/test group overlaps are all zero. This controls related-story
  and author leakage, but the test set represents smaller components and has a
  higher mean rating than training (0.7793 versus 0.7229).
- Split IDs and test predictions are provided without raw story text in adjacent files.
- Dataset card lists Open Use of Data Agreement v1.0:
  https://huggingface.co/datasets/allenai/hippocorpus
- Citation: Sap, Horvitz, Choi, Smith, and Pennebaker (2020), *Recollection versus
  Imagination: Exploring Human Memory and Cognition via Neural Language Models*, ACL.

## Features and model selection

The model uses 12 existing text-derived features, log word count, and 512
SHA-256-hashed word-count features. No fitted vocabulary sees test text. Category
flags come from the existing rule classifier, not ground-truth labels. Recency,
access counts, and interaction scores are excluded because Hippocorpus does not
supply equivalent deployment measurements. Survey responses, demographics,
worker IDs, story type, and pair IDs are not model inputs.

Four XGBoost configurations were evaluated on validation RMSE (depth 2/4 and
minimum child weight 5/20). Early stopping used validation only. The exported
model contains only the selected trees; test data was never used for fitting or
selection. It has not been refit on test data.

Selected depth: 4; minimum child weight: 20; trees: 354.

Full parameters, dependency versions, source checksums, label distribution, and
candidate metrics are in `importance_metrics.json` and the model metadata.

## Integration and limitations

`MemoryCore.with_ml()` pairs `MLFeatureExtractor` with `MLImportanceScorer`.
The scorer retains the `score(features) -> float` contract. Its output feeds the
existing lifecycle manager, persisted record, and retrieval importance signal.
The native XGBoost UBJ model is checksum-checked at load time; no pickle is loaded.
Missing dependencies, incompatible features, or corrupt artifacts fail explicitly.
Records identify the model and target in `source_metadata.importance_model`.

The default `MemoryCore()` still uses the heuristic. Select ML explicitly because
Hippocorpus contains long autobiographical/imagined stories, not short tasks,
preferences, and greetings. Its ratings are skewed toward high importance.
Existing lifecycle thresholds have not been calibrated against this proxy target.
Do not interpret a high model score on a greeting as validated retention value.
No labeled conversational importance or lifecycle-tier evaluation was performed.
An unlabeled integration smoke check produced the following scores:

| Input | ML importance | Assigned tier |
|---|---:|---|
| hello | 0.5806 | working |
| I prefer concise Python explanations. | 0.5632 | long_term |
| Yesterday I celebrated my graduation. | 0.5547 | short_term |

These examples expose the domain mismatch: the model scores a greeting higher
than the preference. The category-based lifecycle rule still keeps it in working
memory. This is why the experimental scorer is explicitly selected, not enabled
as the default. These smoke examples were not used to tune or retrain the model.

## Reproduce

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-ml.txt
.venv/bin/python -m training.train_importance
.venv/bin/python -m unittest -v
```

Run from the repository root. Training downloads the official archive if absent
and verifies the CSV checksum. Raw data and the virtual environment are ignored
by Git; the native model, metadata, reports, and reproducible code are included.
