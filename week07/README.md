# Project 7 — Evaluate and Tune End to End

A chronological, end-to-end classification evaluation on the UCI Bank Marketing dataset. The emphasis is **credible evidence, leakage prevention and diagnostic checks**, not maximizing a single score. The principal finding is that logistic regression and random forest do not demonstrate reliably useful ranking across campaign periods; the final test is reported once, without subsequent tuning.

## Files to submit

```text
p07-<yourname>/
├── analysis.ipynb      # Data audit, chronological split, pipeline searches, validation, one final held-out test
├── checks.ipynb        # Five checks: dummies, single feature, stability, shuffled labels, groups/time
├── report.md           # Required six-section evidence report
├── README.md           # This file
├── data/
│   └── bank-additional-full.csv
└── outputs/             # Reproducible generated result tables
    ├── p07_test_metrics.csv
    ├── p07_test_confusion.csv
    └── p07_test_capacity.csv
```

If your course requires only four primary deliverables, keep the dataset or download instructions and `outputs/` as supporting resources rather than submitting them as separate graded files.

## Dataset

**Source:** UCI Machine Learning Repository, Bank Marketing dataset: <https://archive.ics.uci.edu/dataset/222/bank+marketing>. Use **`bank-additional-full.csv`**, the 41,188-row, 21-column semicolon-delimited file. The data may be packaged in downloadable ZIP archives; extract the actual CSV into the `data/` directory.

The target `y` marks whether a client subscribed to a term deposit. We exclude `duration` to avoid using post-contact information for a pre-contact decision. Nineteen predictors remain. Categorical `unknown` values are handled within fitted pipelines. The complete CSV has an overall positive rate of 11.27% and 12 exactly duplicate full rows. Rows are in historical order; explicit customer IDs are unavailable.

**Path check:** The notebooks must agree on the actual location of the CSV. In the current project instructions, the preferred relative path when notebooks are opened from the Week 07 folder is `data/bank-additional-full.csv`. If your existing `analysis.ipynb` instead uses `../week07/data/bank-additional-full.csv`, that may resolve to the same file when the working directory is `week07`, but consistency is recommended. Verify your notebook's working directory when running it.

## Environment

Recommended Python 3.10+ with compatible versions of:

- numpy
- pandas
- scipy
- scikit-learn (a version supporting `OneHotEncoder(sparse_output=False)`)
- matplotlib
- jupyter / VS Code Jupyter extension

For a local installation in an existing virtual environment:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib notebook
```

For reproducibility, record your exact installed versions (e.g. `python --version` and `python -m pip freeze`) and consider committing a `requirements.txt` captured from the working environment. Dependency versions were not independently verified here.

## Run order

1. Download/extract the actual CSV to `data/bank-additional-full.csv`.
2. Open **`analysis.ipynb`** and run from a fresh kernel. It should audit the data; separate chronological training (60%), development validation (20%), and sealed test (20%); build preprocessing within pipelines; search at least 20 configurations per family using three chronological CV folds; inspect full results; and perform the **single predeclared final held-out evaluation only at the end**.
3. Open **`checks.ipynb`** and run from a fresh kernel. It should independently recreate training data and pipelines; run all dummy baselines, one-feature diagnostic, five expanding time-boundary checks, and **full 20-candidate shuffled-label searches**, and document groups/time limitations. The verification notebook should not access the final test partition.
4. Confirm outputs under `outputs/` and update `report.md` only from saved results. **Do not re-evaluate or tune on the test data**.

Notebook execution order can be kept reproducible by defining shared helper functions in both notebooks or moving them to a small importable module. Avoid relying on in-memory variables defined only in another notebook.

## Predeclared metric, comparison and holdout policy

- **Primary metric:** Average Precision (AP), chosen to assess subscriber ranking under scarce review capacity. Compare with each evaluation fold's positive prevalence.
- **Secondary metrics:** ROC-AUC, log-loss, precision at review capacity, recall at review capacity.
- **Models:** Logistic regression and random forest with preprocessing choices inside each `RandomizedSearchCV` pipeline.
- **Evaluation:** Three chronological CV folds on the earliest 60%; separate later 20% development validation; final latest 20% held out and scored exactly once.
- **Final research challenger:** Original-search selected random forest; no deployment recommended.
- **Locked primary operating point:** top 1,000 ranked test customers; additional capacity summaries at 100, 500 and 2,000.

## Key measured findings

| Statistic | Result |
|---|---:|
| Full dataset | 41,188 rows |
| Training / development validation / final test | 24,712 / 8,238 / 8,238 |
| Training / validation / test positive prevalence | 4.81% / 11.07% / 30.83% |
| Original best CV AP, logistic / forest | 0.05330 / 0.05620 |
| Original CV no-skill AP baseline | ~0.05422 |
| External development-validation AP, logistic / forest | 0.09918 / 0.09684 |
| Development-validation baseline AP | 0.11071 |
| Shuffled-label best CV AP, logistic / forest | 0.05694 / 0.05945 |
| Final-test forest AP / baseline AP | 0.35047 / 0.30833 |
| Final-test forest ROC-AUC | 0.50975 |
| Top-1,000 test selected subscribers / selected non-subscribers | 440 / 560 |
| Top-1,000 test recall | 17.32% |

The final holdout AP exceeds the contemporary prevalence benchmark but ROC-AUC remains only slightly above random. The historical prevalence changes materially, the separate development-validation results were worse than the AP baseline, and the shuffled-label searches outperformed real-label searches. **These results do not support operational deployment or claims of stable superiority over random ranking.**

## Limitations and safeguards

- Chronological order is available but explicit row timestamps and customer IDs are not; repeated-contact leakage cannot be ruled out.
- `duration` is excluded; other inputs still require confirmation of availability at the actual outreach decision moment.
- The observed subscription outcome is not a causal effect of contacting that customer.
- AP is prevalence-sensitive; never interpret the raw CV-to-test AP difference as pure improvement in model quality.
- A single blockwise shuffled-label experiment is a diagnostic, not a formal permutation p-value.
- The original best parameters were compared using the same chronological folds; five alternate chronological folds diagnose stability but do not constitute nested cross-validation.

See `report.md` for the required six-section account of decisions, results, verification checks, and what would change the conclusion.
