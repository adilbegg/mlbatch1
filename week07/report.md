# Project 7 — Evaluate and Tune End to End

## 1. Decision and responsible team

The proposed classifier supports a bank marketing team's decision about which customers to prioritize for term-deposit outreach review when capacity is limited. The team would rank existing records by predicted subscription probability and review the highest-scored customers first. This is a **retrospective outcome-ranking exercise**, not an estimate of the causal benefit of calling a customer. Because the evidence of stable ranking performance is weak, **do not deploy this model for outreach prioritization** based on these results.

## 2. The data: prevalence, missingness, duplicates, groups, and time

We used the UCI Bank Marketing `bank-additional-full.csv` dataset with 41,188 observations. Its original 21 columns include 20 inputs and binary target `y`; `duration` was excluded because call length would not be available at a pre-contact decision point, leaving **19 features** (nine numeric and ten categorical). Overall outcomes: 4,640 subscribers and 36,548 non-subscribers (11.27% positive). There are **12 exactly duplicate full rows**, which do not necessarily identify duplicate customers and were not automatically deleted.

There were **zero literal NaNs** in the source data, but the categorical string `unknown` appeared in `default` (8,597), `education` (1,731), `housing` (990), `loan` (990), `job` (330), and `marital` (80). We treated `unknown` as missing within the categorical pipeline and tuned its imputation strategy (`most_frequent` versus `constant`). Numeric features used median imputation and standard scaling; categorical features used imputation and one-hot encoding, all **fitted within each training fold**, not before splitting. This treatment is a modeling choice; missingness itself may have predictive meaning.

The full dataset has chronological ordering. We preserved row order and split the observations into the earliest 60% (**24,712 training**), next 20% (**8,238 development validation**), and last 20% (**8,238 final test**). Training prevalence was **4.81%** and development validation prevalence was **11.07%**. Once accessed during the single locked final evaluation, final-test prevalence was **30.83%** (2,540 subscribers). The changing base rates are strong evidence of temporal prevalence shift and make historical performance difficult to extrapolate. The dataset does not provide an explicit customer identifier; repeated contacts cannot be isolated reliably between folds. Published ordering is an approximation of time; exact per-row timestamps are not available here. No claim of group-independence is made.

## 3. Primary metric and justified baseline

**Primary metric declared before fitting: Average Precision (AP).** The operational objective is to find eventual subscribers within limited review capacity, and positive cases were rare in the earlier historical data. AP assesses how well subscribers appear near the top of the ranked list over multiple precision–recall operating points, which is more closely connected to this use case than overall accuracy.

For a constant-score/no-skill ranking, AP equals evaluation-set prevalence: approximately **0.05422** averaged across the three original chronological CV validation folds, **0.11071** on development validation, and **0.30833** on final test. These baselines necessarily differ when prevalence changes. Secondary metrics were **ROC-AUC** (ranking discrimination), **log-loss** (probability quality), **precision at review capacity** (fraction of selected customers who subscribed), and **recall at capacity** (fraction of all subscribers captured). The no-skill log-loss benchmark used probabilities equal to training prevalence rather than using unknown future outcomes.

## 4. Search design and model distinguishability

Two end-to-end pipeline families were compared using `RandomizedSearchCV`, **20 sampled configurations per family**, **three expanding-window chronological CV folds**, and `scoring='average_precision'`. All transformations were inside the searched pipeline. Logistic regression search included log-uniform `C` (0.001–100), class weights (`None`/`balanced`), and categorical imputation strategy. Random forest search included number of trees (60–120), tree depth, log-uniform fractional minimum leaf size (0.001–0.03), `max_features`, and the same categorical imputation choice. This satisfies tuning of preprocessing as well as model settings.

| Family | Best mean CV AP | CV fold SD | Best CV fold AP scores | Approx. SE | Configurations within 1 SE |
|---|---:|---:|---|---:|---:|
| Logistic regression | 0.05330 | 0.01130 | 0.03959, 0.06715, 0.05311 | 0.0065 | 20/20 |
| Random forest | 0.05620 | 0.01400 | 0.04118, 0.07479, 0.05250 | 0.0081 | 20/20 |

The paired forest-minus-logistic differences were **0.00159, 0.00763, −0.00061** (mean **0.00287**, sample SD **0.00427**). The gap is smaller than its across-fold spread. The two model families **cannot be distinguished convincingly** on this evidence, despite forest's slightly larger CV mean. All sampled configurations in each family were within one estimated standard error of the best result. The standard-error calculation on three time-dependent folds is descriptive, not a formal inferential interval.

The CV-selected logistic model used `C=67.322489`, balanced class weights, and constant categorical imputation. The selected forest used maximum depth 15, `max_features=1.0`, minimum leaf fraction 0.0283368, 116 trees, and most-frequent categorical imputation. Some alternative forests showed very large training-validation AP gaps; hyperparameter flexibility did not yield robust out-of-time improvement.

## 5. Result: CV spread, external validation, locked test, and capacity

The separate chronological development-validation set yielded:

| Approach | Validation AP | ROC-AUC | Log-loss |
|---|---:|---:|---:|
| Training-prevalence baseline | 0.11071 | 0.50000 | 0.37981 |
| Logistic regression | 0.09918 | 0.48891 | 0.85911 |
| Random forest | 0.09684 | 0.47351 | 0.35077 |

Both models **failed to outperform the AP baseline on later development data**. The forest had better log-loss than the historical-prevalence baseline but poor ranking. Given these findings, operational deployment was rejected. To demonstrate the course's final holdout procedure, the random forest was predeclared as the **final research challenger** based on its slightly higher original CV AP; review budgets **100, 500, 1,000, and 2,000** and primary operating point **top 1,000** were fixed before the final test evaluation.

The single locked final held-out evaluation on **8,238 latest records** yielded:

| Approach | Test AP | ROC-AUC | Log-loss |
|---|---:|---:|---:|
| Training-prevalence baseline | 0.30833 | 0.50000 | 0.96986 |
| Random forest | **0.35047** | **0.50975** | **0.89836** |

The reported **forest CV AP of 0.05620 versus test AP of 0.35047** produces a raw difference of **+0.29427**. This is **not evidence of a comparably large improvement in ranking skill**: the underlying positive prevalence moved from ~4–6% in CV to 30.83% on test, and AP depends on prevalence. Relative to each evaluation period's no-skill baseline, the original three-fold forest CV gain was roughly **+0.00198 AP**, while final test gain was **+0.04214 AP**. Test ROC-AUC remained near random at 0.50975. The difference in results is consistent with severe temporal distribution shift and a lack of stable discrimination. The baseline's test log-loss was especially poor because the training prevalence of 4.81% severely underpredicted later observed prevalence.

At the **top-1,000** operating point, the confusion matrix was **TN 5,138; FP 560; FN 2,100; TP 440**. Precision was **0.4400** and recall **0.17323**. Here a false positive means a selected historical non-subscriber, not necessarily an economically wasteful outreach; a true positive means a selected historical subscriber, not proof that a new call caused the subscription.

| Review budget | Subscribers in model top-K | Selected non-subscribers | Precision | Recall | Expected subscribers in random K |
|---:|---:|---:|---:|---:|---:|
| 100 | 41 | 59 | 0.410 | 0.0161 | 30.83 |
| 500 | 223 | 277 | 0.446 | 0.0878 | 154.16 |
| 1,000 | 440 | 560 | 0.440 | 0.1732 | 308.33 |
| 2,000 | 778 | 1,222 | 0.389 | 0.3063 | 616.65 |

The expected random counts are retrospective calculations from **test prevalence**, not deployable prospective predictions or observed random-selection trials. Model selection or tuning was not repeated after evaluating this test set.

## 6. Five verification checks and what would change the conclusion

**Check 1 — All dummy strategies.** The original three-fold dummy mean APs were: stratified **0.05512**, and prior, most-frequent, uniform, always-negative, and always-positive each **0.05422**. The stratified result fluctuates with randomness and is not evidence of a customer-level signal. Real-label logistic CV AP **0.05330** failed to exceed the constant-score baseline. Forest CV AP **0.05620** exceeded it by only **0.00198**.

**Check 2 — Best single feature.** The leading single-feature logistic regression was `age`, mean AP **0.05933** (fold APs **0.04387, 0.06715, 0.06697**), above the full logistic and forest scores. `age` is ordinarily knowable before a marketing call and is not inherently leakage, but the age-band relationships varied over time. For example, observed under-30 subscription rates were **4.40%, 6.25%, 8.86%** in successive examined folds. The 60+ bands contained only 43, 38, and 42 records, producing unstable rate estimates. Nineteen separate feature screenings also create a winner's-selection effect. The observation warrants investigation and cannot establish stable future utility.

**Check 3 — Split stability.** Because time order must be preserved, five **expanding chronological validation boundaries** were used instead of random split seeds. Mean AP / sample SD: logistic **0.05260 / 0.01391**, forest **0.05544 / 0.01774**. Original three-fold reported CV APs were approximately **0.05067 SD** (logistic) and **0.04268 SD** (forest) above the respective five-fold averages, but individual performance was unstable. In fold 5, baseline AP was **0.05464**, logistic **0.04713**, and forest **0.04063**. This check assesses time-boundary sensitivity, not five random-seed repetitions, and does not independently reselect hyperparameters within every fold.

**Check 4 — Full shuffled-label search.** The complete randomized searches were rerun, preserving 20 candidates, three chronological CV folds, pipeline fitting and preprocessing for each family. Labels were permuted within chronological blocks `[0, 6178, 12356, 18534, 24712]`, preserving **1,188 positive cases** overall and each block's prevalence; **9.13%** of labels changed value, which is expected for rare positive labels. The best shuffled-label APs were **0.05694 logistic** and **0.05945 forest**, compared with real-label APs **0.05330** and **0.05620**. Both shuffled searches exceeded their real-label counterparts. In one permutation, this is a serious warning against treating the small original AP as compelling signal; it is **not proof of leakage**, not a formal p-value, and blockwise permutation retains broad time-prevalence patterns. Selected shuffled scores may also reflect selection over 20 candidates.

**Check 5 — Groups and time.** Chronological ordering was respected using an earlier 60% development-training partition, a subsequent 20% validation partition, a latest 20% final holdout, and expanding-window CV. There is no explicit customer ID, so we cannot guarantee separation of repeated customers; 12 identical full rows are not reliable identifiers. Per-row dates and true deployment-time feature availability cannot be completely verified from these data.

**Assumption that could invalidate deployment:** Historical subscription outcomes and available predictors must represent the population, decision moment, and customer-level ranking relationship encountered in future campaigns. The observed class-prevalence shift, weak ROC-AUC, adverse development-validation performance, and shuffled-label findings challenge that assumption. Before any operational deployment, obtain prospectively timestamped data with stable customer IDs; establish that every input exists before the contact decision; use strictly later, customer-group-separated evaluation; assess AP lift and precision at realistic call capacity across multiple successive campaigns; and test incremental contact effects using a randomized outreach experiment. **Do not deploy** unless the model repeatedly and materially beats practical baselines and passes data-availability and fairness reviews.

### Scope of verification

All numerical outcomes above were taken from the executed notebook outputs supplied for this project. They have not been independently reproduced from the complete source dataset and current notebook files in this report-generation environment.
