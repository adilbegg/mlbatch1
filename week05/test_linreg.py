import numpy as np

from sklearn.linear_model import LinearRegression
from linreg import MyLinearRegression

rng = np.random.default_rng(42)

X = rng.normal(size=(100, 3))

true_coefficients = np.array([2.0, -3.0, 5.0])

y = (
    10
    + X @ true_coefficients
    + rng.normal(0, 0.1, size=100)
)

my_model = MyLinearRegression()

sklearn_model = LinearRegression()

my_model.fit(X, y)
sklearn_model.fit(X, y)

print("My coefficients:", my_model.coef_)
print("Sklearn coefficients:", sklearn_model.coef_)

print("My intercept:", my_model.intercept_)
print("Sklearn intercept:", sklearn_model.intercept_)

coefficient_difference = np.max(
    np.abs(my_model.coef_ - sklearn_model.coef_)
)

intercept_difference = np.abs(
    my_model.intercept_ - sklearn_model.intercept_
)

prediction_difference = np.max(
    np.abs(
        my_model.predict(X)
        - sklearn_model.predict(X)
    )
)

print("Maximum coefficient difference:", coefficient_difference)
print("Intercept difference:", intercept_difference)
print("Maximum prediction difference:", prediction_difference)

assert coefficient_difference < 1e-10
assert intercept_difference < 1e-10
assert prediction_difference < 1e-10

print("OLS reference test: PASSED")

print("\nTest Ridge against Scikit-learn")

from sklearn.linear_model import Ridge

alphas = [0.1, 1.0, 10.0,  100.0]

for alpha in alphas:

    my_ridge = MyLinearRegression(alpha=alpha, penalty='l2')

    sklearn_ridge = Ridge(alpha=alpha, solver='cholesky')

    my_ridge.fit(X, y)
    sklearn_ridge.fit(X, y)

    difference = np.max(np.abs(my_ridge.coef_ - sklearn_ridge.coef_))

    print(f"Alpha = {alpha}")

    print(f"Maximum difference = {difference:.15e}")

    assert difference < 1e-10

print("All Ridge Test: PASSED")

for alpha in alphas:

    my_ridge = MyLinearRegression(alpha=alpha)
    sklearn_ridge = Ridge(
        alpha=alpha,
        solver="cholesky"
    )

    my_ridge.fit(X, y)
    sklearn_ridge.fit(X, y)

    intercept_diff = abs(
        my_ridge.intercept_ - sklearn_ridge.intercept_
    )

    print(
        f"Alpha {alpha}: "
        f"Intercept difference = {intercept_diff:.15e}"
    )

    assert intercept_diff < 1e-10

#Verify Ridge shrinks coefficients
ols = MyLinearRegression(alpha=0)
ridge = MyLinearRegression(alpha=100,penalty='l2')

ols.fit(X, y)
ridge.fit(X, y)

print("OLS Coefficients:", ols.coef_)
print("Ridge Coefficieents:", ridge.coef_)

print("OLS coefficient norm:", np.linalg.norm(ols.coef_))
print("Ridge coefficient norm:", np.linalg.norm(ridge.coef_))


print("\nTest Lasso against Scikit-learn")

from sklearn.linear_model import Lasso

lasso_alphas = [0.001, 0.01, 0.1]

for alpha in lasso_alphas:

    my_lasso = MyLinearRegression(
        alpha=alpha,
        penalty="l1"
    )

    sklearn_lasso = Lasso(
        alpha=alpha,
        fit_intercept=True,
        max_iter=100000,
        tol=1e-12,
        selection="cyclic"
    )

    my_lasso.fit(X, y)
    sklearn_lasso.fit(X, y)

    coef_difference = np.max(
        np.abs(my_lasso.coef_ - sklearn_lasso.coef_)
    )

    intercept_difference = abs(
        my_lasso.intercept_ - sklearn_lasso.intercept_
    )

    print(f"\nAlpha: {alpha}")
    print(f"Coefficient difference: {coef_difference:.12e}")
    print(f"Intercept difference: {intercept_difference:.12e}")
    print(f"My iterations: {my_lasso.n_iter_}")

    assert coef_difference < 1e-4
    assert intercept_difference < 1e-4

print("\nAll Lasso comparisons: PASSED")

ols = MyLinearRegression(alpha=0, penalty="l2")
lasso_zero = MyLinearRegression(alpha=0, penalty="l1")

ols.fit(X, y)
lasso_zero.fit(X, y)

assert np.array_equal(ols.coef_, lasso_zero.coef_)
assert ols.intercept_ == lasso_zero.intercept_

print("Test 1 --> Zero-alpha tests: PASSED")

# Test 2: Regression without intercept

my_no_intercept = MyLinearRegression(
    alpha=0,
    fit_intercept=False
)

sklearn_no_intercept = LinearRegression(
    fit_intercept=False
)

# Both models must use the SAME dataset
my_no_intercept.fit(X, y)
sklearn_no_intercept.fit(X, y)

# Compare coefficients
coef_difference = np.max(
    np.abs(
        my_no_intercept.coef_
        - sklearn_no_intercept.coef_
    )
)

print("My intercept:", my_no_intercept.intercept_)
print("Sklearn intercept:", sklearn_no_intercept.intercept_)
print("Coefficient difference:", coef_difference)

# Verify zero intercept
assert my_no_intercept.intercept_ == 0

# Verify coefficients match sklearn
assert np.allclose(
    my_no_intercept.coef_,
    sklearn_no_intercept.coef_,
    atol=1e-10,
    rtol=0
)

# Verify coefficients differ from OLS with intercept
assert not np.allclose(
    my_no_intercept.coef_,
    ols.coef_
)

print("Test 2 --> No-intercept tests: PASSED")

rng = np.random.default_rng(42)

x = rng.normal(size=100)

X_collinear = np.column_stack([x, 2 * x])
y_collinear = 5 + 3 * x

collinear_model = MyLinearRegression(alpha=0)
collinear_model.fit(X_collinear, y_collinear)

assert np.isfinite(collinear_model.coef_).all()
assert np.allclose(
    collinear_model.predict(X_collinear),
    y_collinear,
    atol=1e-10,
    rtol=0
)

print("Test 3 --> Collinearity test: PASSED")

rng = np.random.default_rng(42)

X_wide = rng.normal(size=(10, 20))
y_wide = rng.normal(size=10)

wide_ols = MyLinearRegression(alpha=0)
wide_ridge = MyLinearRegression(alpha=1)

wide_ols.fit(X_wide, y_wide)
wide_ridge.fit(X_wide, y_wide)

assert len(wide_ols.coef_) == 20
assert len(wide_ridge.coef_) == 20

assert np.isfinite(wide_ols.coef_).all()
assert np.isfinite(wide_ridge.coef_).all()

print("Test 4 --> More-features-than-rows tests: PASSED")

def expect_value_error(X_bad, y_bad):

    try:
        MyLinearRegression().fit(X_bad, y_bad)

    except ValueError:
        return True

    return False


assert expect_value_error(
    [[1], [2], [3]],
    [1, 2]
)

assert expect_value_error(
    [],
    []
)

assert expect_value_error(
    [[1], [np.nan], [3]],
    [1, 2, 3]
)

assert expect_value_error(
    [[1], [2], [3]],
    [1, np.nan, 3]
)

assert expect_value_error(
    np.ones((5, 2, 2)),
    np.ones(5)
)

print("Test 5 --> Input-validation tests: PASSED")

rng = np.random.default_rng(42)

X_check = rng.normal(size=(100, 3))
y_check = (
    5
    + X_check @ np.array([2, -3, 4])
    + rng.normal(size=100)
)

ols_check = MyLinearRegression(alpha=0)
ols_check.fit(X_check, y_check)

residuals = y_check - ols_check.predict(X_check)

residual_sum = np.sum(residuals)
orthogonality = X_check.T @ residuals

print("Residual sum:", residual_sum)
print("Feature-residual products:", orthogonality)

assert abs(residual_sum) < 1e-10
assert np.max(np.abs(orthogonality)) < 1e-10

print("Test 6 --> OLS mathematical-property tests: PASSED")

# Test 7: Residual-report diagnostics

import numpy as np

# Create a reproducible synthetic dataset
rng = np.random.default_rng(42)

X_check = rng.normal(size=(100, 3))

y_check = (
    5
    + X_check @ np.array([2, -3, 4])
    + rng.normal(0, 1, size=100)
)

# Fit our regression model
diagnostic_model = MyLinearRegression(alpha=0)
diagnostic_model.fit(X_check, y_check)

# Generate the residual report
report = diagnostic_model.residual_report(
    X_check,
    y_check
)

# Display diagnostic results
for assumption, result in report.items():
    print(assumption, ":", result)

# Check that all four diagnostic sections exist
assert set(report.keys()) == {
    "linearity",
    "independence",
    "homoscedasticity",
    "normality"
}

# Check that the diagnostic statistics are valid
assert np.isfinite(
    report["independence"]["durbin_watson"]
)

assert 0 <= report["homoscedasticity"]["p_value"] <= 1
assert 0 <= report["normality"]["p_value"] <= 1

print("Residual-report tests: PASSED")

print("\nTEST 8: OLS on different dataset shapes")

rng = np.random.default_rng(42)

shapes = [
    (100, 1),
    (100, 3),
    (10, 20)
]

for n_rows, n_features in shapes:

    X_shape = rng.normal(
        size=(n_rows, n_features)
    )

    y_shape = rng.normal(size=n_rows)

    my_ols = MyLinearRegression(alpha=0)
    sklearn_ols = LinearRegression()

    my_ols.fit(X_shape, y_shape)
    sklearn_ols.fit(X_shape, y_shape)

    difference = np.max(
        np.abs(my_ols.coef_ - sklearn_ols.coef_)
    )

    print(
        f"Shape {X_shape.shape}: "
        f"Maximum difference = {difference:.12e}"
    )

    assert difference < 1e-10

print("TEST 8: PASSED")

print("\nTEST 9: R2 score comparison")

rng = np.random.default_rng(42)

X_score = rng.normal(size=(100, 3))

y_score = (
    5
    + X_score @ np.array([2, -3, 4])
    + rng.normal(size=100)
)

my_score_model = MyLinearRegression()
sklearn_score_model = LinearRegression()

my_score_model.fit(X_score, y_score)
sklearn_score_model.fit(X_score, y_score)

my_r2 = my_score_model.score(X_score, y_score)
sklearn_r2 = sklearn_score_model.score(X_score, y_score)

difference = abs(my_r2 - sklearn_r2)

print("My R2:", my_r2)
print("Sklearn R2:", sklearn_r2)
print("Difference:", difference)

assert difference < 1e-12

print("TEST 9: PASSED")

print("\nTEST 10: Specific error messages")

error_cases = [
    (
        [[1], [2], [3]],
        [1, 2],
        "same number of rows"
    ),
    (
        [],
        [],
        "empty"
    ),
    (
        [[1], [np.nan], [3]],
        [1, 2, 3],
        "NaN"
    ),
    (
        [[1], [2], [3]],
        [1, np.nan, 3],
        "NaN"
    ),
    (
        np.ones((5, 2, 2)),
        np.ones(5),
        "1-D or 2-D"
    )
]

for X_bad, y_bad, expected_message in error_cases:

    try:
        MyLinearRegression().fit(X_bad, y_bad)

        raise AssertionError(
            "An expected ValueError was not raised."
        )

    except ValueError as error:
        message = str(error)

        assert expected_message.lower() in message.lower()

        print("Correct error:", message)

print("TEST 10: PASSED")

print("\nTEST 11: Lasso tolerance sensitivity")

rng = np.random.default_rng(42)

X_tol = rng.normal(size=(200, 5))

y_tol = (
    3
    + X_tol @ np.array([2, -1, 0, 4, 0])
    + rng.normal(0, 0.5, size=200)
)

tolerances = [1e-3, 1e-6, 1e-10, 1e-12]

reference = MyLinearRegression(
    alpha=0.01,
    penalty="l1"
)
reference.fit(X_tol, y_tol)

for tolerance in tolerances:

    sklearn_lasso = Lasso(
        alpha=0.01,
        tol=tolerance,
        max_iter=100000,
        selection="cyclic"
    )

    sklearn_lasso.fit(X_tol, y_tol)

    difference = np.max(
        np.abs(
            reference.coef_ - sklearn_lasso.coef_
        )
    )

    print(
        f"Tolerance = {tolerance:.0e}, "
        f"Difference = {difference:.12e}"
    )