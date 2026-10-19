# Project 4 — The Preprocessing Pipeline

A reusable, leak-resistant preprocessing pipeline for predicting NYC Airbnb listing prices using scikit-learn.

## Dataset

NYC Airbnb Open Data (2019), containing 48,895 rows and 16 original columns.

Source: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

Download `AB_NYC_2019.csv` and place it inside the `data/` folder.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run the Pipeline

```python
from pipeline import build_pipeline

model = build_pipeline()  # Returns an unfitted pipeline
```

## Project Files

- `pipeline.py` — Importable scikit-learn pipeline
- `audit.ipynb` — Dataset audit, missingness and scaling evidence
- `leak_tests.ipynb` — Functional tests and leakage experiments
- `justification.md` — Evidence-based preprocessing decisions
- `requirements.txt` — Python dependencies

## Reproducibility

Run `audit.ipynb` and `leak_tests.ipynb` from top to bottom using the project environment. Train/test splitting and randomized cross-validation use a fixed random seed of 42.

## Baseline Results

| Metric | Result |
|---|---:|
| Test MAE | $70.89 |
| Test R² | 0.149 |
| Five-fold CV MAE | $72.27 |
| Transformed features | 105 |

The pipeline contains numeric imputation, missingness indicators, evidence-based scaling, low-cardinality one-hot encoding, rare-category grouping and Ridge regression.