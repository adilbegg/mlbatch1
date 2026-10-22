# Project 6 — Cost-Based Operating Threshold (R6)

## 1. Decision context, data, and limits

The UCI **Bank Marketing** (`bank-additional-full.csv`) data contain 41,188 records describing customers contacted during a marketing campaign. The outcome `y=1` means **term-deposit subscription**. The observed positive rate is **11.2654%**, so the task is imbalanced. We excluded `duration` because call length is not known before the call. The remaining design uses 19 original predictors, including numerical and categorical features.

A seeded, stratified random split produced **24,712 training**, **8,238 validation**, and **8,238 test** observations. The preprocessing transformations (imputation, scaling, one-hot encoding) are fitted only within the training pipeline. The test set was reserved until costs, model choice, and threshold-selection rules were locked.

This is an **illustrative classification-cost exercise**. It does **not** estimate the causal effect or incremental profitability of contacting an individual. Everyone in these historical records was part of a contacted population; counterfactual outcomes without a call are unavailable. The selected rule also ignores contact capacity and other operational constraints.

## 2. Costs specified before threshold optimization

- **False positive (FP): €2 per case.** We prioritize a customer whom the observed outcome records as a non-subscriber. This hypothetical amount represents staff time and communication resources spent on an unsuccessful marketing contact. It is a fixed educational assumption, not a measured bank cost.
- **False negative (FN): €15 per case.** We fail to prioritize a customer who subscribed in the observed contacted dataset. This hypothetical amount represents foregone net value from overlooking a prospective subscriber. It is **not** a causal estimate of lost profit, since subscription absent our prioritization strategy is unobserved.

True positives and true negatives incur zero *error cost* in this simplified matrix. Costs are assumed constant across customers. The total metric is `2 × FP + 15 × FN` euros; no costs were changed after looking at validation/test predictions.

## 3. Theoretical decision rule

For a calibrated probability `p = P(y=1 | x)`, the expected error cost of flagging is `(1−p) × €2`; that of skipping is `p × €15`. Flag when the former is no greater than the latter. Therefore:

\[
t_{theory}=\frac{C_{FP}}{C_{FP}+C_{FN}}=\frac{2}{2+15}=\mathbf{0.1176470588}.
\]

This is a threshold of **11.76%**, not the conventional 50%. It is defensible because the assumed consequence of missing a subscriber is much larger than that of a false alarm. The rule relies on sufficiently calibrated probabilities; it is not appropriate to treat uncalibrated class-weighted scores as literal risks.

## 4. Validation-only threshold sweep

On the **8,238 validation examples**, we evaluated all distinct class-1 probabilities plus 0 and 1 (**8,147 candidate thresholds**). Ties in minimum cost were resolved by selecting the cutoff closest to the theoretical 11.76% threshold. No test labels were involved.

| Validation rule | Threshold | TN | FP | FN | TP | Flagged | Cost (€) | Cost/customer (€) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unweighted, default | 0.500000 | 7,195 | 115 | 714 | 214 | 329 | 10,940 | 1.3280 |
| Unweighted, theoretical | 0.117647 | 6,236 | 1,074 | 348 | 580 | 1,654 | 7,368 | 0.8944 |
| Unweighted, validation minimum | **0.114958** | 6,208 | 1,102 | 341 | 587 | 1,689 | **7,319** | **0.8884** |

The empirical minimum improves the theoretical cost by just **€49** on validation, equivalent to about **0.67% of the minimum**. A total of **300 candidate thresholds from 0.1098 to 0.1614** were within **1% of minimum validation cost**. These values reveal a relatively broad/flat near-optimal region, not a unique economically meaningful cutoff to eight decimal places. The 1% region is a sensitivity description, not a confidence interval.

![Measured validation cost curve, with default, theoretical and empirical thresholds](outputs/validation_cost_curve.png)

**Pre-test decision:** Retain **0.117647** as the *primary* operational threshold because it derives from predeclared costs, the unweighted model is comparatively well calibrated, and its validation cost is almost indistinguishable from the empirical minimum. Retain **0.114958** as the validation-tuned alternative; compare both without retuning after seeing test outcomes.

## 5. Held-out test results: exact confusion counts and consequences

On the untouched **8,238-example test split**, there were **7,310 actual non-subscribers** and **928 subscribers**.

| Fitted model and rule | Threshold | TN | FP | FN | TP | Flagged | Precision | Recall | Specificity | F1 | Cost (€) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unweighted, default | 0.5000 | 7,213 | 97 | 711 | 217 | 314 | 0.6911 | 0.2338 | 0.9867 | 0.3494 | 10,859 |
| **Unweighted, primary theoretical** | **0.1176** | **6,273** | **1,037** | **326** | **602** | **1,639** | **0.3673** | **0.6487** | **0.8581** | **0.4690** | **6,964** |
| Unweighted, validation-selected | 0.1150 | 6,246 | 1,064 | 321 | 607 | 1,671 | 0.3633 | 0.6541 | 0.8544 | 0.4671 | 6,943 |
| Balanced, default | 0.5000 | 6,269 | 1,041 | 326 | 602 | 1,643 | 0.3664 | 0.6487 | 0.8576 | 0.4683 | 6,972 |

The primary decision reduces the hypothetical test error cost by **€3,895 (35.87%)** relative to the default. It identifies **385 more observed subscribers**, reducing missed subscribers from 711 to 326, while flagging **1,325 additional customers**, including **940 additional false positives**. In plain terms, the bank team would need to prioritize substantially more customer contacts to find more of the people associated with subscription in these records. This is a resource trade-off, not proof that additional calls would *cause* 385 additional subscriptions.

The empirical 11.50% cutoff costs only **€21 less** than the declared 11.76% primary rule on test. **We do not switch thresholds in response to that test result.** Both results are reported because the assignment requests a formula-based and a validation-selected rule.

![Test confusion matrix at default threshold](outputs/test_confusion_default.png)

![Test confusion matrix at primary cost-derived threshold](outputs/test_confusion_primary.png)

![Test confusion matrix for balanced classifier at 0.50](outputs/test_confusion_balanced.png)

## 6. Calibration and comparison of imbalance strategies (R5/R7)

The constant-probability baseline predicts the **training prevalence** for each example and achieves held-out **log-loss 0.3520**. The always-negative classifier's accuracy is approximately **88.73%**, yet it has zero positive recall.

| Strategy | Precision | Recall | Log-loss | Test ECE (10 equal-width buckets) | Test cost (€) |
|---|---:|---:|---:|---:|---:|
| Unweighted, 0.50 | 0.6911 | 0.2338 | **0.2717** | **0.0152** | 10,859 |
| Unweighted, validation-selected 0.1150 | 0.3633 | 0.6541 | **0.2717** | **0.0152** | **6,943** |
| Balanced, 0.50 | 0.3664 | 0.6487 | 0.5155 | 0.2785 | 6,972 |
| **Unweighted, primary 0.1176** | 0.3673 | 0.6487 | **0.2717** | **0.0152** | **6,964** |

The threshold alone does **not** change predicted probabilities, so both unweighted strategies have identical log-loss and ECE. Class-weighted training shifts the probabilities upward: average test probability is about **0.3911**, compared with the actual positive rate of approximately **0.1126**. The unweighted average probability is **0.1120**. This aligns with the much lower test ECE of the unweighted model and the reliability-table evidence. Calibration tables with ten buckets are in `analysis.ipynb` and the exported CSV files.

On test, the unweighted model's ECE increased from validation **0.0065** to **0.0152**. The balanced model's ECE was **0.2807** on validation and **0.2785** on test. A smaller ECE is preferable but depends on bucket definitions; in particular, high-probability buckets may contain relatively few customers.

**Recommendation:** For these stated classification-error costs, choose the **unweighted, calibrated-probability model at the predeclared 11.76% threshold**. It delivers materially better recall and lower assumed cost than 0.5 without the substantial calibration and log-loss deterioration of balanced training. The 11.50% validation choice is a close alternative, not grounds for changing the locked primary rule.

## 7. Limitations

1. **Noncausal marketing outcomes:** Customers were historically contacted; the classifier is not an uplift or treatment-effect model.
2. **Assumed costs:** €2 and €15 are pedagogical constants, not bank-provided marginal-profit estimates.
3. **Capacity constraints:** There is no explicit maximum number of calls, fatigue constraint, or per-customer contact cost model beyond the simplified error matrix.
4. **Random split:** A chronological holdout might better reflect future campaigns and distribution shift.
5. **Calibration diagnostics:** Ten-bin ECE can hide within-bin errors and is unstable in tiny bins; low ECE alone does not guarantee a well-specified probability model.
6. **Model implementation:** Large mixed-type real-data fits use scikit-learn pipelines. The from-scratch implementation is validated separately with synthetic examples, gradient checks, and reference reconciliation.
