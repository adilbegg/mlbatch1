# Project 6 — Logistic Regression from Scratch

Binary classification, numerical reconciliation, calibrated probabilities, and cost-based decision thresholds. Developed for Week 06 / Project 6, with a UCI Bank Marketing real-data experiment.

## Quick start (from `week06/`)

```bash
python -m pip install numpy pandas matplotlib scipy scikit-learn jupyter
python test_logistic.py
jupyter notebook analysis.ipynb
# In Jupyter: Kernel → Restart & Run All
```

**Dataset:** Download UCI [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), extract **`bank-additional-full.csv`** from `bank-additional.zip` and place it in `week06/data/`. The notebook reads **`data/bank-additional-full.csv`** using `sep=';'`. Data is **not** bundled in this documentation package. Reproducibility also requires keeping the notebook's seeded splits and fitting the preprocessing pipeline on training data only.

## Required files

```text
week06/
├── logistic.py           # Custom LogisticClassifier
├── test_logistic.py      # Numerical stability, gradients, training, invalid inputs
├── analysis.ipynb        # Synthetic checks, sklearn reconciliation, real data
├── reconciliation.md     # Four-stage reference comparison
├── threshold.md          # Costs, calibration, threshold search, held-out outcome
├── README.md
├── data/
│   └── bank-additional-full.csv   # User downloads from UCI
└── outputs/              # Optional reproducibility artifacts from notebook
    ├── validation_threshold_sweep.csv
    ├── final_test_results.csv
    ├── strategy_comparison.csv
    ├── test_unweighted_calibration.csv
    ├── test_balanced_calibration.csv
    ├── validation_cost_curve.png
    ├── test_confusion_default.png
    ├── test_confusion_primary.png
    └── test_confusion_balanced.png
```

## What the custom classifier implements

`LogisticClassifier(lr=0.1, steps=10000, l2=0.0, class_weight=None, tol=1e-9)` supports `fit(X, y)`, `predict_proba(X)` (a **1D array of positive-class probabilities**, not scikit-learn's two-column format), `predict(X, threshold=0.5)`, and `score(X, y)` (accuracy). After `fit`, it exposes `w` (intercept first), `intercept_`, `coef_`, `n_iter_`, `converged_at`, and `loss_history_`.

Implementation details: stable piecewise sigmoid, numerically stable logit-domain binary loss (`np.logaddexp`), batch gradient descent, L2 regularization of **feature weights only**, class-weight balancing normalized to mean one, parameter-change convergence detection, and input validation. `converged_at=None` indicates that the specified parameter-change stopping condition was not reached within the allotted steps. This stopping rule is not a proof of mathematical optimality.

The test script exercises extreme logits of ±1,000, five gradient vectors (including zero and far-from-optimum), weighted/L2 finite differences, synthetic classifier training, and input errors. The latest local revision was reported by the project author to pass after adding single-class and invalid-input checks; the copy previously supplied for audit predates those latest fixes.

## Numerical reconciliation (R3)

The custom solver (batch gradient descent) and scikit-learn (`lbfgs`) optimize a matched unweighted L2-regularized logistic objective on a 500-row synthetic dataset. With our mean-loss objective and `l2=0.01`, match `C = 1/(500*0.01) = 0.2`.

| Stage | Maximum coefficient gap |
|---|---:|
| scikit-learn defaults | 1.183776e-01 |
| Match regularization | 4.238989e-05 |
| Explicitly confirm intercept convention | 4.238989e-05 |
| Tighten tolerances | **5.017710e-10** |

The last small nonzero gap reflects finite numerical optimization using different solvers and stopping rules; see [`reconciliation.md`](reconciliation.md) for coefficient, intercept and probability discrepancies.

## Real-world dataset and leakage controls

UCI Bank Marketing: **41,188 rows**, initially 20 predictors, **19 after removing `duration`**, positive rate **11.2654%**. The `duration` feature is excluded because the duration of a call is not available when prioritizing calls. Stratified seeded splits: **24,712 train**, **8,238 validation**, **8,238 test**. Numeric/categorical preprocessing is fitted inside scikit-learn pipelines on training rows; **no scaling before splitting, and no threshold choice using the test set**.

The real-data pipelines use scikit-learn `LogisticRegression`, not `LogisticClassifier`, because mixed-type preprocessing and large-scale optimized fitting are handled there. The custom algorithm is independently verified and reconciled on synthetic data. The baseline that always predicts negative has approximately **88.73% accuracy** but **zero positive recall**. The base-rate probability baseline's held-out log-loss is **0.3520**.

## Main test results

Assumed error costs, declared before optimizing the validation threshold: **FP = €2**, **FN = €15**. The theoretical operating threshold is **2/(2+15) = 0.117647**; validation sweep minimum is **0.114958**. The theoretical rule was preselected as primary because its validation cost was within €49 of the minimum, within a broad near-minimal range.

| Decision rule | Recall | Precision | Test log-loss | Test ECE | Assumed test cost |
|---|---:|---:|---:|---:|---:|
| Unweighted, 0.50 | 0.2338 | 0.6911 | 0.2717 | 0.0152 | €10,859 |
| **Unweighted, 0.1176 (primary)** | **0.6487** | 0.3673 | **0.2717** | **0.0152** | **€6,964** |
| Unweighted, 0.1150 (validation optimum) | 0.6541 | 0.3633 | 0.2717 | 0.0152 | €6,943 |
| Balanced, 0.50 | 0.6487 | 0.3664 | 0.5155 | 0.2785 | €6,972 |

Primary-rule assumed cost falls **€3,895 (35.87%)** relative to default, while flagging **1,325 more customers** and identifying **385 more observed subscribers** in this retrospective sample. The balanced model achieves similar decisions at threshold 0.5 but substantially worse calibration and log-loss.

**Important interpretation:** These are retrospective classification results under hypothetical fixed costs, **not a causal estimate of the extra subscriptions or profits generated by making additional calls**. The data are observational and were randomly split rather than tested on a future chronological campaign. Consult [`threshold.md`](threshold.md) for the explicit cost rationale, both threshold calculations, confusion matrices, calibration discussion and limitations.

## Reproduce and submit

Run tests, then restart and run the notebook **in order**. Confirm that `outputs/` CSV files and figures correspond to the notebook's final metrics; commit `logistic.py`, `test_logistic.py`, `analysis.ipynb`, the three Markdown files, and any required figures/CSV evidence. The original source dataset should be credited to UCI; follow instructor/repository guidance on whether to commit the CSV or document its download procedure.
