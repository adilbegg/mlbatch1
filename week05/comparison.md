# Project 5 — Numerical Agreement with scikit-learn

## Purpose and reproducibility

`MyLinearRegression` in `linreg.py` implements OLS, Ridge and Lasso without using scikit-learn to solve coefficients. `test_linreg.py` compares it with scikit-learn on seeded synthetic datasets. Run from `week05/`:

```bash
python3 test_linreg.py
```

The numerical values below are **measured outputs from the user's recorded local test runs**, not theoretical error bounds. On another machine or BLAS/LAPACK build, differences in the last decimal places can vary; the supplied script may reproduce slightly different machine-precision values. The submitted test file contains **32 written `assert` statements** (some execute repeatedly inside loops). It prints individual PASS messages but not a single aggregate assertion counter.

## 1. Ordinary least squares

For the reference dataset generated with `np.random.default_rng(42)`, 100 rows, three independent standard-normal predictors, true coefficient vector `[2, -3, 5]`, intercept 10 and Gaussian noise with standard deviation 0.1:

| Quantity compared | Actual absolute difference | Required tolerance | Result |
|---|---:|---:|---|
| Maximum coefficient difference | `4.440892098500626e-16` | `< 1e-10` | PASS |
| Intercept difference | `0.0` | `< 1e-10` | PASS |
| Maximum training-prediction difference | `3.552713678800501e-15` | `< 1e-10` | PASS |

The custom estimates were `[2.0005595, -2.99302072, 4.98025007]`, with intercept `10.008767734454858`, matching the reference to displayed precision.

**Additional data shapes** (independent seeded random targets):

| Matrix shape (rows × predictors) | Maximum absolute coefficient difference | Tolerance | Result |
|---|---:|---:|---|
| `(100, 1)` | `1.387778780781e-17` | `< 1e-10` | PASS |
| `(100, 3)` | `3.469446951954e-17` | `< 1e-10` | PASS |
| `(10, 20)` | `3.608224830032e-16` | `< 1e-10` | PASS |

For a separate seeded score-check dataset, both implementations returned `R² = 0.9521810091736465`, difference **0**, satisfying `< 1e-12`.

## 2. Ridge regression

Both implementations use the scikit-learn Ridge objective `||y − Xw||² + α||w||²`. The custom version centers predictors and the target for fitting an intercept, then solves `(XcᵀXc + αI)w = Xcᵀyc` using `np.linalg.solve`, **not** `np.linalg.inv`.

| α | Maximum coefficient difference | Intercept difference | Tolerance | Result |
|---:|---:|---:|---:|---|
| 0.1 | `8.881784197001252e-16` | `0` | `< 1e-10` | PASS |
| 1.0 | `8.881784197001252e-16` | `0` | `< 1e-10` | PASS |
| 10.0 | `6.661338147750939e-16` | `0` | `< 1e-10` | PASS |
| 100.0 | `2.220446049250313e-16` | `0` | `< 1e-10` | PASS |

For this synthetic dataset, the Euclidean coefficient norm decreased from **6.1451852776443845** for OLS to **2.5822066612147627** for Ridge at α=100. This demonstrates shrinkage on this dataset; it does not establish that Ridge improves all datasets' test performance.

## 3. Lasso regression

Both implementations use `1/(2n) ||y − Xw||² + α||w||₁`. The custom implementation uses cyclic coordinate descent and soft thresholding, with a coefficient-change stopping threshold of `1e-12` and up to 100,000 cycles. For comparison, scikit-learn Lasso was fitted with `selection='cyclic'`, `max_iter=100000`, and `tol=1e-12`.

| α | Maximum coefficient difference | Intercept difference | Custom cycles | Required tolerance | Result |
|---:|---:|---:|---:|---:|---|
| 0.001 | `3.863576125696e-14` | `1.776356839400e-15` | 11 | `< 1e-4` | PASS |
| 0.01 | `3.930189507173e-14` | `0` | 11 | `< 1e-4` | PASS |
| 0.1 | `3.796962744218e-14` | `1.776356839400e-15` | 11 | `< 1e-4` | PASS |

**Why use `1e-4` rather than promise `1e-8` everywhere?** Lasso is iterative and its stopping criterion differs from scikit-learn's dual-gap checks. Conditioning, predictor correlation and tolerances can affect the final coefficient values. The near-`1e-14` agreement above is an **observed result on these data**, not a universal Lasso guarantee.

### Convergence-tolerance sensitivity

For an additional seeded `(200, 5)` synthetic dataset, we compared scikit-learn's Lasso coefficients at several tolerances with the same tightly converged custom fit (`alpha=0.01`):

| scikit-learn tolerance | Measured maximum absolute difference |
|---:|---:|
| `1e-3` | `1.261483924488e-4` |
| `1e-6` | `8.125268946557e-8` |
| `1e-10` | `3.306688256544e-12` |
| `1e-12` | `2.442490654175e-13` |

Tighter tolerances improved agreement on this example. These four numerical outputs are observations; the sensitivity loop does **not** itself contain an assertion or guarantee monotonic improvement on all datasets.

On a distinct sparse-recovery example with true coefficients `[4, 0, 0, -3, 0]`, the custom Lasso estimated approximately `[3.80912665, 0, 0, -2.80406792, 0]`. It set the three irrelevant-feature coefficients to zero on this sample.

## 4. Edge cases and analytical properties

The executable suite additionally checks:

- `alpha=0` yields the same OLS solution for L1 and L2 settings.
- `fit_intercept=False` yields zero intercept and coefficients matching scikit-learn, distinct from the intercept-fitted case in the test data.
- Perfectly collinear predictors produce finite least-squares coefficients and correct fitted predictions.
- Wide matrices (`p > n`) yield finite OLS and positive-α Ridge fits.
- Mismatched sample counts, empty arrays, NaNs in either input, and three-dimensional feature matrices raise specific `ValueError` messages.
- OLS residuals sum approximately to zero and are approximately orthogonal to the predictor columns: residual sum `8.881784197001252e-16`, feature inner products approximately `[-1.04114120e-14, 1.39023094e-14, 8.81309305e-14]`.
- `residual_report()` produces diagnostic entries for linearity, independence, variance constancy and normality. It is a set of indicators, **not a proof** that classical regression assumptions hold.

### Known limits of the submitted implementation

- For a multiple-feature fitted model, a single prediction row must be shaped `(1, p)` rather than a flat list of length `p`.
- The Lasso solver's stored iteration count does not currently issue a warning if its hard iteration limit is reached.
- The squared-fitted-value/residual correlation in `residual_report()` is a limited nonlinearity indicator and may be undefined when fitted values are constant. Durbin–Watson requires meaningful observation order; BP and JB diagnostics require appropriate statistical assumptions for formal inference.
- Scikit-learn is used in the **tests and real-data analysis**, not to solve the custom implementation's coefficients.

## 5. Alpha and intercept conventions

- Ridge custom/scikit-learn: `||y − Xw||² + α||w||²` (no `1/n` factor on loss).
- Lasso custom/scikit-learn: `1/(2n)||y − Xw||² + α||w||₁`.
- Some textbooks use different normalizations; the same numerical α is **not** a matched penalty strength across the two methods.
- For fitted intercepts, `X` and `y` are centered, coefficients are fitted without a penalized bias column, and the intercept is recovered as `mean(y) − mean(X) @ coef_`.
- `alpha=0` uses `np.linalg.lstsq`, including for singular or wide matrices.

See `report.md` for the separate real-world evaluation and `analysis.ipynb` for original plots. Actual observed differences are documented here to avoid substituting qualitative PASS claims for numerical evidence.
