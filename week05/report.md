# Project 5 — Linear Regression from Scratch: Empirical Analysis and Evaluation

## 1. Dataset choice and model appropriateness

I used the NYC Airbnb 2019 listings dataset, originally containing **48,895 observations and 16 columns**. After excluding 11 listings with non-positive nightly price, **48,884 observations** remained. The outcome was nightly listing price in USD. The dataset includes at least seven numerical features (latitude, longitude, minimum nights, number of reviews, reviews per month, calculated host listings count, and availability in the next 365 days), as well as categorical location and room-type information. The data were divided once into **39,107 training** and **9,777 held-out test** observations, with an 80/20 split and random state 42.

This is a useful but challenging linear-regression setting: prices are positive, highly right-skewed and affected by location, property characteristics and missing attributes not represented in the dataset. Accordingly, a linear model is an interpretable baseline rather than a complete structural or causal pricing model. The Week 04 preprocessing pipeline handles missing values, feature transformations and categorical encoding inside a scikit-learn `Pipeline`; these operations are learned separately inside training folds during cross-validation.

The custom `MyLinearRegression` class is implemented and assessed separately in `linreg.py` and `test_linreg.py`. **Real-data experiments in `analysis.ipynb` use scikit-learn Ridge and Lasso** with the reused Week 04 preprocessor. No claim is made that the custom estimator powered the final real-data pipeline.

## 2. Agreement with reference implementations

The NumPy-based implementation supports unpenalized OLS using `np.linalg.lstsq`, Ridge using `np.linalg.solve`, and Lasso using cyclic coordinate descent with soft thresholding. The intercept is handled by centering when enabled and is not penalized. An explicit matrix inverse is never used.

### OLS on a three-feature synthetic dataset

| Quantity | Maximum absolute difference from scikit-learn |
|---|---:|
| Coefficients | 4.440892098500626e-16 |
| Intercept | 0 |
| Predictions | 3.552713678800501e-15 |

Additional OLS shapes were tested: `(100, 1)` had maximum coefficient difference `1.387778780781e-17`; `(100, 3)` had `3.469446951954e-17`; `(10, 20)` had `3.608224830032e-16`. Custom and scikit-learn R² scores were identical to displayed precision (`0.9521810091736465`; difference zero).

### Ridge agreement

| Alpha | Maximum coefficient difference | Intercept difference |
|---:|---:|---:|
| 0.1 | 8.881784197001252e-16 | 0 |
| 1 | 8.881784197001252e-16 | 0 |
| 10 | 6.661338147750939e-16 | 0 |
| 100 | 2.220446049250313e-16 | 0 |

At alpha 100, the coefficient norm declined from `6.1451852776443845` (OLS) to `2.5822066612147627` (Ridge) on the synthetic dataset.

### Lasso agreement

| Alpha | Maximum coefficient difference | Intercept difference | Custom iterations |
|---:|---:|---:|---:|
| 0.001 | 3.863576125696e-14 | 1.776356839400e-15 | 11 |
| 0.01 | 3.930189507173e-14 | 0 | 11 |
| 0.1 | 3.796962744218e-14 | 1.776356839400e-15 | 11 |

OLS and Ridge comparisons satisfy the `1e-10` tolerance; Lasso comparisons satisfy `1e-4`. Unlike closed-form OLS and Ridge, Lasso is iterative, so its agreement depends on convergence criteria, feature conditioning and tolerance. When scikit-learn Lasso's tolerance was tightened from `1e-3` to `1e-12`, the maximum difference relative to the custom tightly converged reference decreased in this experiment from `1.261483924488e-4` to `2.442490654175e-13`. This behavior is observed on the supplied synthetic data and is not claimed to hold identically for every dataset.

Tests also covered zero alpha, no intercept, collinear features, more features than rows, non-finite and malformed inputs, informative error messages, and OLS residual sum and feature orthogonality. `residual_report()` provided statistics for four diagnostic categories. See `test_linreg.py` for executable assertions and their exact coverage.

## 3. Implementation conventions

**OLS:** Solves the least-squares problem with `np.linalg.lstsq`, including rank-deficient and wide designs without explicit inversion.

**Ridge:** Minimizes `||y - Xw||² + alpha * ||w||²`, following scikit-learn's Ridge alpha convention. For alpha greater than zero, the coefficients solve `(Xc.T @ Xc + alpha*I) w = Xc.T @ yc`, using `np.linalg.solve`.

**Lasso:** Minimizes `(1/(2n)) ||y - Xw||² + alpha * ||w||₁`, following scikit-learn's Lasso convention, using cyclic coordinate descent and soft thresholding `S(z, alpha) = sign(z) max(|z|-alpha, 0)`. **Ridge and Lasso alpha values are not directly comparable** because their objectives and scaling differ.

With `fit_intercept=True`, the predictors and outcome are centered during fitting and the intercept is subsequently recovered as `mean(y) - mean(X) @ coefficients`; it is not included in the penalty. With `fit_intercept=False`, the intercept is zero. The real-data pipeline fits preprocessing on training folds only. Model selection and correction factors were obtained from training observations; the test set was evaluated only after the final choice was fixed.

## 4. Four-assumption audit: numbers, visuals, limitations and consequences

The initial Ridge model used alpha 10, selected by five-fold training-only cross-validation. Its best mean fold R² was `0.1183148407993174`; pooled out-of-fold R² was `0.11111844245994507`, MAE was **$72.3374** and residual standard deviation was approximately **$234.929**. The fold-averaged and pooled R² values are not expected to be identical.

**Linearity / mean specification.** Residuals against fitted prices and the binned-residual plot showed a systematic U-shaped pattern. For example, the lowest fitted-price decile had mean predicted price ~$25.29 and mean residual **+$37.48**; a middle-to-upper bin around $185.96 had mean residual **−$19.03**; the highest decile around $307.14 had mean residual **+$29.81**. This is evidence of a misspecified conditional mean, not a definitive proof of a particular nonlinear functional form. Consequence: systematic under- and overprediction in different price ranges. See `analysis.ipynb` for residual-versus-fitted and binned-residual plots.

**Independence.** A Kruskal–Wallis comparison of residual distributions across borough groups gave statistic `963.6678452343331` and p-value `2.6667603445793938e-207`, indicating differences across groups under the test's assumptions. **This is not an independence test**: it does not prove spatial autocorrelation. Because the data are cross-sectional and CSV row order is arbitrary, a Durbin–Watson statistic based on row order would be difficult to interpret. Consequence if spatial dependence exists: standard errors and uncertainty calculations assuming independent observations can be misleading. A suitable spatial-neighbour definition and spatial autocorrelation diagnostic would be required to assess this rigorously.

**Homoscedasticity.** Absolute residuals versus fitted prices showed uneven error magnitudes and influential extreme listings. The exploratory Breusch–Pagan test using out-of-fold residuals returned LM statistic `43.010113220132034`, p-value `5.445762978658616e-11`. This is evidence of nonconstant residual variance, subject to the limitation that conventional BP calibration assumes a suitable single-model regression residual setting. Consequence: classical constant-variance standard errors and inference may be unreliable; large-price listings may behave differently from typical listings.

**Normality.** Original dollar residuals were extremely right-skewed: Jarque–Bera statistic `873603485.1592863`, displayed p-value `0.0` (numerical underflow), skewness `22.42243688426243` and excess kurtosis `730.8352467499418`. The histogram and Q–Q plot showed a very long positive tail. Consequence: exact normal-theory small-sample inference is inappropriate without qualification, although normal residuals are not required simply to compute OLS coefficients or make predictions.

These diagnostics are not independent proof of every classical OLS assumption; in particular, many are exploratory diagnostics on regularized, out-of-fold predictions. Plots and numerical evidence should be read together.

## 5. Intervention, before/after results and final model evaluation

The intervention was to predict `log1p(price)` and invert with `expm1`. This reduced the influence of extremely expensive listings while learning. The uncorrected transformation decreased out-of-fold dollar MAE, but created positive mean dollar residuals (systematic underprediction), consistent with retransformation bias. A **Duan global smearing correction** was therefore evaluated, with its factor learned from each fold's training subset only and applied to that fold's validation predictions.

| Training-only OOF experiment | R² | MAE (USD) | Mean signed residual (USD) | Median absolute error (USD) |
|---|---:|---:|---:|---:|
| Original dollar Ridge, alpha 10 | 0.1111 | 72.3374 | 0.0054 | 43.2686 |
| Log-target Ridge, alpha 10, no correction | 0.1038 | 60.5950 | 24.0732 | 26.1138 |
| Log-target Ridge + smearing, alpha 10 | 0.1180 | 66.4533 | 1.9471 | 34.4608 |

The log transform substantially reduced residual skewness **on the transformed scale** to `1.4058314322776644` and excess kurtosis to `7.439741120721758`. It did **not** restore normality (Jarque–Bera statistic `103071.70626069205`, displayed p-value zero) or constant variance (exploratory BP statistic `174.2205477258657`, p-value `8.859743115102237e-40`). These log-scale diagnostics are not directly comparable in magnitude to dollar-scale variance tests. Smearing improved dollar-scale average calibration, but did not automatically fix log-scale statistical assumptions. Binned residual plots still showed price-dependent bias at the high end.

Five-fold tuning of log-target Ridge with smearing using training data only selected **alpha 0.1** by the preselected primary criterion, pooled out-of-fold R². Its OOF R² was `0.11845541034263607` (approximately `0.1185`) and MAE was **$66.4963**. Alpha 1 had nearly identical R². The final model was trained on all **39,107** training rows with smearing factor **1.1694586946211147**. The 9,777-row test set was evaluated once after the model was fixed.

| Evaluation | R² | MAE (USD) |
|---|---:|---:|
| Mean baseline: training | 0.000000 | 92.950793 |
| Mean baseline: test | -0.000112 | 91.281764 |
| Final model: training | 0.120591 | 66.243796 |
| Final model: CV, pooled OOF | 0.118455 | 66.496261 |
| Final model: test | **0.152399** | **65.498822** |

Final training RMSE was **$233.6736**, and final test RMSE was **$184.2132**. Test MAE improved by approximately **28.2%** versus the mean baseline. However, the low test R² means much price variation remains unexplained. The final test R² being higher than training/CV R² does not imply test leakage; differences in the difficulty and distribution of observations may contribute, but this was not established by a separate distribution audit.

## 6. Coefficients, units, multicollinearity and Ridge–Lasso comparison

The final Ridge model has **105 transformed features**, with intercept `4.163399737687689` on the `log1p(price)` scale. Coefficients belong to *transformed* predictors, not automatically to raw columns or dollars. For a one-unit change in a transformed predictor with coefficient beta, predicted `(price + 1)` is multiplied by `exp(beta)`, holding the other transformed variables fixed; the percentage change in `(price + 1)` is `100 * (exp(beta)-1)`. When a feature is standardized, one unit means one training standard deviation; for scaled availability it means the full 0-to-1 scaled range; for logged numeric variables it means one unit of their transformed-and-scaled representation. The global smearing factor cancels in a ratio of corrected `(price + 1)` predictions when clipping is inactive.

The model's room-type coefficients were `+0.615381` for Entire home/apartment, `-0.113991` for Private room and `-0.501390` for Shared room. The model-implied contrast between Entire home/apartment and Private room is `0.7293727899059426` log points, corresponding to **approximately 107.38% higher predicted `(price + 1)`**, conditional on all other modeled information. This is a conditional model association, not a causal claim. Because the one-hot encoding retains all categories, category-specific coefficients should be interpreted as *contrasts*, not isolated absolute premiums.

For seven original numerical features after training-only median imputation, VIFs were:

| Feature | VIF |
|---|---:|
| reviews_per_month | 1.511487 |
| number_of_reviews | 1.498158 |
| availability_365 | 1.136332 |
| calculated_host_listings_count | 1.094731 |
| longitude | 1.053726 |
| minimum_nights | 1.051637 |
| latitude | 1.009741 |

These indicate limited linear redundancy **in this seven-variable, median-imputed specification**. They do not measure collinearity across the entire 105-column transformed matrix. In fact, all-category one-hot indicators alongside an intercept can create exact dependencies. VIF above 10 is a conventional warning heuristic, not a universal proof of invalidity. Ridge stabilizes fitted predictions under collinearity, but individual coefficients may still be difficult to interpret.

At exploratory alpha **0.01**, scikit-learn Lasso on the same training set and log-target preprocessing retained **9 of 105** features, setting **96** coefficients to exactly zero. The retained features and coefficients were: Entire home/apartment `0.761443`; Manhattan `0.252505`; availability_365 `0.192336`; standardized longitude `-0.100099`; Williamsburg `0.068598`; log minimum nights `-0.064272`; review missingness indicator `0.040291`; Midtown `0.025763`; and log number of reviews `-0.019650`. Lasso's training R² was `0.0786`, and training MAE was **$62.9819**; Ridge plus smearing had training R² `0.1206` and training MAE **$66.2438**. These numbers **do not establish relative generalization performance**: the Lasso comparison did not include smearing or tuned regularization, and neither was selected using the final test set. Ridge/Lasso alpha values also cannot be interpreted as comparable penalty strengths. The zero coefficients indicate selection under this fitted specification, not that excluded predictors are economically irrelevant.

## 7. Overall judgment and limitations

The custom implementations reproduced scikit-learn closely on synthetic reference tests, including multiple shapes, collinearity and invalid inputs. The real-data pipeline followed a training-only fitting and selection workflow and improved held-out MAE over the mean baseline. Nonetheless, the modest test R², nonlinear binned-residual patterns, very heavy original error tails, evidence of heteroscedasticity and possible geographical structure limit strong claims about accuracy and inference.

The logarithmic outcome transformation and smearing correction were useful practical interventions: the former improved typical errors and residual shape; the latter improved average dollar calibration and squared-error performance relative to the uncorrected log model. Neither removed all diagnostic concerns. Lasso produced a sparse representation but did not demonstrate that the retained nine features form the uniquely correct explanatory model.

**Judgment:** A regularized linear model is a valuable, interpretable baseline and can provide useful approximate pricing estimates; it is **not sufficient as a precise or causal model of NYC Airbnb prices**. Further work could explore nonlinear functions, interactions, stronger spatial modeling and uncertainty estimates that account for heteroscedasticity and potential dependence. Such work is outside the present finalized held-out evaluation; the test set should not be reused to tune it.

### Interpretation of the intervention evidence

The before/after numbers above are derived from *training-only out-of-fold predictions*. They are legitimate diagnostics for the intervention but are **not** a same-observations before/after comparison of the two models on the held-out test set, which was reserved for the single finalized model. The final test R² and MAE below therefore describe the **selected** smearing-corrected model, not the effect size of the intervention in an independently tested head-to-head trial. This limitation prevents overstating causal claims about the model change.

## Supporting artifacts

- `linreg.py`: custom OLS, Ridge and Lasso implementations and residual diagnostics.
- `test_linreg.py`: numerical reference, edge-case and diagnostic tests.
- `comparison.md`: detailed numerical agreement table and Lasso tolerance discussion.
- `analysis.ipynb`: preprocessing, training-only cross-validation, residual plots, intervention, held-out evaluation, VIFs and coefficient interpretation.
- `outputs/model_performance.csv`, `outputs/numerical_vif.csv`, `outputs/ridge_lasso_coefficients.csv`, `outputs/lasso_selected_features.csv`: generated analysis tables.
