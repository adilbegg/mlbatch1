# Project 6 — Reconciliation with scikit-learn (R3)

## Objective and setup

The goal is to determine whether the implementation in `logistic.py` and scikit-learn's `LogisticRegression` minimize the same mathematical objective, then quantify the remaining numerical differences. Comparisons use the **same seeded synthetic dataset** of 500 observations and two predictors, generated with `np.random.default_rng(42)` and binary outcomes sampled from logits `-0.5 + 1.5*x1 - 1.0*x2`.

Our `LogisticClassifier` uses batch gradient descent. The reference uses scikit-learn's `lbfgs` solver. In the compared experiment, both fit a **free, unpenalized intercept**, use no class weights, and penalize the two feature coefficients with L2.

### Loss and matching convention

Our objective, with `l2 = λ`, is:

\[
J(b,w)=\frac{1}{n}\sum_{i=1}^n [\operatorname{logaddexp}(0,z_i)-y_i z_i]+\frac{\lambda}{2}\lVert w\rVert_2^2,\qquad z_i=b+x_i^\top w.
\]

Under the scikit-learn `lbfgs` objective and unweighted training convention used here, the equivalent penalty in **mean-loss units** is `||w||² / (2*C*n)`. Hence `λ = 1/(C*n)` and `C = 1/(n*λ)`. With `n=500` and `λ=0.01`, the matched scikit-learn setting is **`C=0.2`**. The intercept is excluded from both L2 penalties. This mapping is solver/objective-specific; it is not a universal identity for arbitrary class-weight or normalization conventions.

## Measured reconciliation

The table reports **maximum absolute** differences on the same 500 training examples. “Coefficient” compares the two feature coefficients; “intercept” compares the scalar intercepts; “probability” compares class-1 predicted probabilities.

| Stage | Max coefficient difference | Intercept difference | Max probability difference |
|---|---:|---:|---:|
| 1. scikit-learn defaults (`C=1`) | 1.183776e-01 | 1.363074e-02 | 2.639529e-02 |
| 2. Matched penalty (`C=0.2`) | 4.238989e-05 | 6.167160e-06 | 3.544558e-05 |
| 3. Explicit `fit_intercept=True` | 4.238989e-05 | 6.167160e-06 | 3.544558e-05 |
| 4. Tightened tolerances | **5.017710e-10** | **8.265931e-10** | **5.870741e-10** |

Stage 3 is unchanged because `fit_intercept=True` was already the default and `lbfgs` does not penalize the intercept. Its purpose is to **confirm** the convention, not mechanically reduce the gap.

For Stage 4, custom gradient descent used `lr=0.1`, `steps=50000`, `l2=0.01`, and `tol=1e-11`. It recorded convergence at **2,192 steps**. Scikit-learn used `C=0.2`, `solver='lbfgs'`, `fit_intercept=True`, `tol=1e-12`, and `max_iter=20000`, finishing in **8 L-BFGS iterations**.

![Maximum coefficient disagreement across reconciliation stages](outputs/reconciliation_curve.png)

> To display this figure in GitHub, save the reconciliation plot from `analysis.ipynb` as `outputs/reconciliation_curve.png`. This document's table contains all measured numerical values even if the plot is absent.

## Why the final difference is not exactly zero

The models optimize the same smooth, penalized loss but do so through **different iterative algorithms**. Our implementation takes repeated fixed-learning-rate gradient steps and stops based on the largest parameter update. L-BFGS uses a quasi-Newton direction and its own convergence test. Finite tolerances, floating-point arithmetic, and the algorithms' stopping conditions can produce residual differences. A coefficient gap of about `5.0e-10` is exceptionally small and is supported by the explicit matching procedure; it is **not** an assertion of bit-for-bit identity or a claim that both used the same solver.

This is a numerical agreement experiment on synthetic data, not proof that every input, weight configuration, or real-world dataset will match to the same precision. Separately, five central-difference gradient checks (including the all-zero parameter vector and a far-from-optimum vector) had worst discrepancies between `4.71e-12` and `1.30e-10`. The weighted-plus-L2 gradient check had worst discrepancy about `1.63e-11`. These support the correctness of the implemented gradient relative to its own loss.

## Reproducing the comparison

From `week06/`, run the tests and notebook:

```bash
python test_logistic.py
jupyter notebook analysis.ipynb
```

In the notebook, execute the synthetic training and four-stage reconciliation cells in order. **Do not tune tolerances or rewrite the table based on the real-data test split.**
