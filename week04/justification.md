# Project 4: The Preprocessing Pipeline
**Dataset:** NYC Airbnb Open Data (2019)  
**Task:** Predict nightly listing prices using a reusable, leak-safe scikit-learn pipeline.

## 1. Dataset and Messiness Requirements

The NYC Airbnb 2019 dataset contains 48,895 observations and 16 columns. It provides a realistic preprocessing problem because it combines missing values, numerical outliers, categorical features and high-cardinality variables.

All four dataset requirements were verified using Python:

| Requirement | Evidence |
|---|---|
| At least 1,000 rows and 8 columns | 48,895 rows and 16 columns |
| Numeric feature with at least 3% missing values | `reviews_per_month`: 10,052 missing (20.56%) |
| Categorical feature with fewer than 10 categories | `room_type`: 3; `neighbourhood_group`: 5 |
| Categorical feature with more than 30 categories | `neighbourhood`: 221 |

The target variable is `price`. Eleven zero-price observations were excluded because the modeling task concerns positive nightly listing prices, leaving 48,884 observations. This was a target-validity decision rather than a fitted feature transformation.

## 2. Dataset Audit

The audit was conducted before implementing the transformers. Numerical outliers were screened using the 1.5 × IQR rule. These counts identify unusual values, not necessarily invalid observations.

| Column | Type | Missing % | Distinct | Range / Top Categories | IQR Outliers | Treatment |
|---|---|---:|---:|---|---:|---|
| id | int | 0 | 48,895 | 2,539–36,487,245 | 0 | Drop identifier |
| name | text | 0.03 | 47,905 | Listing titles | N/A | Exclude free text |
| host_id | int | 0 | 37,457 | 2,438–274,321,313 | 1,526 | Drop identifier |
| host_name | text | 0.04 | 11,452 | Michael, David, Sonder (NYC) | N/A | Exclude text |
| neighbourhood_group | text | 0 | 5 | Manhattan, Brooklyn, Queens | N/A | One-hot |
| neighbourhood | text | 0 | 221 | Williamsburg, Bedford-Stuyvesant, Harlem | N/A | Rare-group + one-hot |
| latitude | float | 0 | 19,048 | 40.49979–40.91306 | 425 | StandardScaler |
| longitude | float | 0 | 14,718 | −74.24442–−73.71299 | 2,833 | StandardScaler |
| room_type | text | 0 | 3 | Entire home, Private, Shared | N/A | One-hot |
| price | int | 0 | 674 | 0–10,000 | 2,972 | Target |
| minimum_nights | int | 0 | 109 | 1–1,250 | 6,607 | Log + StandardScaler |
| number_of_reviews | int | 0 | 394 | 0–629 | 6,021 | Log + StandardScaler |
| last_review | text/date | 20.56 | 1,764 | Recent review dates | N/A | Drop raw date |
| reviews_per_month | float | 20.56 | 937 | 0.01–58.5 | 1,793 | Zero impute, indicator, log + scale |
| calculated_host_listings_count | int | 0 | 47 | 1–327 | 7,081 | Log + RobustScaler |
| availability_365 | int | 0 | 366 | 0–365 | 0 | MinMaxScaler |

## 3. Preprocessing Decisions and Evidence

### 3.1 Missing-data mechanisms

**`reviews_per_month`:** I observed that all 10,052 missing entries occur when `number_of_reviews` is zero. Among listings with at least one review, none of the monthly review values are missing. This is strong evidence against MCAR. The pattern is structurally explained by the absence of reviews and is compatible with MAR conditional on observed review count.

I therefore replaced missing monthly review frequencies with zero and retained a missingness indicator. The indicator is redundant with the zero-review condition, but makes the missingness explicit.

The distribution changed as follows:

| Statistic | Before | After zero imputation |
|---|---:|---:|
| Mean | 1.373 | 1.091 |
| Median | 0.720 | 0.370 |
| Standard deviation | 1.680 | 1.597 |
| Skewness | 3.130 | 3.301 |

The zero imputation lowered the mean and median because it included listings with no reviews.

**`last_review`:** The 10,052 missing dates occur in exactly the same rows as missing monthly review frequencies. I therefore treated the missingness as structurally related to zero recorded reviews. The raw date was excluded from the baseline model because date parsing and recency calculation would require additional assumptions. Its missing indicator was not retained separately, since review count already captures the corresponding missingness pattern.

**`name`:** Only 16 values were missing. Listings with missing names had mean minimum stays of 64.88 nights compared with 7.01 for non-missing names. However, the medians were 2 and 3 nights, respectively, suggesting extreme values influence the mean. The median review counts were 0 versus 5. An association with observed variables is plausible, making MAR a working hypothesis, but the small sample prevents a confident classification. The free-text field and its indicator were excluded.

**`host_name`:** Only 21 values were missing. Listings with missing host names had a mean review count of 42.14, versus 23.27 for other listings, but the medians were close at 6 and 5. This suggests the average is influenced by higher-review properties. MAR is plausible but not proven, and an MNAR mechanism cannot be ruled out. Host-name text and its indicator were excluded.

Mean, median and standard deviation are not inherently meaningful for raw listing names, host names or unparsed dates. For those columns, missingness rates, group comparisons and the exclusion rationale were reported instead.

### 3.2 Numerical transformations and scaling

Numerical treatment was chosen from observed skewness, outliers and the interpretation of each feature.

| Feature | Evidence | Decision |
|---|---|---|
| latitude | Skew 0.237; bounded geographic coordinate | StandardScaler |
| longitude | Skew 1.284; geographic coordinate with uneven location density | StandardScaler |
| minimum_nights | Skew reduced from 21.827 to 1.499 | log1p + StandardScaler |
| number_of_reviews | Skew reduced from 3.691 to 0.363 | log1p + StandardScaler |
| reviews_per_month | Observed-value skew reduced from 3.130 to 0.719 | Zero imputation + log1p + StandardScaler |
| calculated_host_listings_count | Skew reduced from 7.933 to 3.280; tail remains substantial | log1p + RobustScaler |
| availability_365 | 35.86% zero; bounded 0–365; median 45 | MinMaxScaler |

Log transformations compress long positive tails without deleting observations. RobustScaler was selected for host listing counts because considerable asymmetry remains after log transformation. MinMaxScaler was chosen for availability to retain a proportional scale.

These transformations improve numerical representation, but do not guarantee a normal distribution or better predictive accuracy.

### 3.3 Categorical encoding

`room_type` and `neighbourhood_group` were one-hot encoded. Neither variable has a justified numerical ordering, so ordinal encoding would introduce artificial relationships. `handle_unknown="ignore"` allows unfamiliar categories during prediction.

The `neighbourhood` column has 221 unique values, including 53 neighbourhoods with fewer than 10 listings, 123 with fewer than 50 and 156 with fewer than 100.

I therefore selected `OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist")`. This groups rare neighbourhoods while retaining separate features for frequent locations.

Category frequencies are learned within training folds through the pipeline. The trade-off is reduced location specificity for uncommon neighbourhoods.

### 3.4 Pipeline and model validation

The preprocessing steps were combined in one `ColumnTransformer`, followed by Ridge regression in a scikit-learn `Pipeline`.

The pipeline operates on raw predictor DataFrames and drops unselected columns internally. It produced 105 transformed features, including the monthly-review missingness indicator.

The pipeline successfully handled unfamiliar categories and fully missing numerical columns. Five-fold cross-validation completed successfully.

Observed baseline results:

- Held-out test MAE: **$70.89**
- Held-out test R²: **0.149**
- Initial five-fold training CV MAE: **$72.27**, standard deviation **$1.96**

These figures describe predictive performance, not proof of causal effects. The relatively low R² indicates that substantial variation in Airbnb listing prices remains unexplained.

## 4. Leakage Experiments

### Experiment 1: Preprocessing before cross-validation

The correct approach fitted the entire preprocessing pipeline separately within each training fold.

The deliberately incorrect approach fitted the preprocessor on the complete cross-validation dataset before fitting Ridge regression across the folds. This allowed validation-fold feature distributions to influence preprocessing.

Both approaches used the same five shuffled folds.

| Fold | Correct MAE | Leaked MAE |
|---|---:|---:|
| 1 | 72.0793 | 71.9842 |
| 2 | 69.6509 | 69.6262 |
| 3 | 72.2236 | 72.4575 |
| 4 | 71.6194 | 71.5927 |
| 5 | 76.1120 | 75.9877 |
| **Mean** | **72.3370** | **72.3297** |

The leaked preprocessing reduced average MAE by approximately $0.0074. This difference is negligible in practical terms and the leaked approach did not outperform in every fold.

The important finding is methodological: preprocessing learned from validation observations before evaluation. Even when the score changes very little, the validation procedure is contaminated.

### Experiment 2: Unseen categories

A one-hot encoder configured with `handle_unknown="error"` raised a ValueError when encountering `Unknown Room Type`.

With `handle_unknown="ignore"`, the encoder returned `[[0, 0, 0]]` without an exception.

The complete fitted pipeline also successfully predicted a row with previously unseen neighbourhood and room-type values, demonstrating its robustness to unfamiliar categories.

## 5. Improvements With More Time or Data

Given additional time, I would compare multiple rare-category thresholds using nested or held-out validation, investigate more flexible models for nonlinear price relationships, and examine whether separate geographic effects improve performance.

I would also investigate the usefulness of review recency using a date-derived feature created inside the pipeline, with an explicit reference date and missingness treatment. Feature engineering from names or amenities might improve predictions if reliable information were available.

Additional years of listing data would enable temporal validation and provide evidence about how well the pipeline generalizes beyond the 2019 snapshot.

## 6. Limitations and Failure Modes

The pipeline is leak-resistant during ordinary training and cross-validation, but it does not handle every possible input.

**Missing required columns:** If a required raw feature such as `latitude` is absent entirely, `ColumnTransformer` will raise an error. This differs from a present column containing only NaN values, which was tested successfully.

**Invalid feature values:** Negative values in variables passed through `log1p` may cause undefined transformations or invalid numerical outputs. The pipeline does not automatically validate their plausible ranges.

**Unexpected input types:** Numeric columns supplied as arbitrary text may cause imputation or scaling errors. Date strings and other unselected features are excluded rather than parsed.

**Rare-category information loss:** An unseen neighbourhood may be grouped with unrelated infrequent neighbourhoods, even when its economic or geographic characteristics differ substantially.

**Indicator stability:** `SimpleImputer(add_indicator=True)` creates indicators for features observed missing during fitting. If monthly reviews have no missing values during fitting, that indicator will not be generated, although the pipeline can still operate with the fitted output structure.

**Data drift and prediction limits:** Changes in rental markets, neighbourhood prices and listing practices may reduce performance. The fitted model also does not guarantee strictly positive price predictions.

These limitations show that preprocessing protects against certain classes of data leakage and input failure, but does not replace ongoing data validation and monitoring.