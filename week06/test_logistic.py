import numpy as np

from logistic import LogisticClassifier


# Test several input scores
logits = np.array([-1000, -10, -2, 0, 2, 10, 1000])

probabilities = LogisticClassifier.sigmoid(logits)

print("Logits:")
print(logits)

print("\nProbabilities:")
print(probabilities)

# All values must be valid numbers
assert np.isfinite(probabilities).all()

# Probabilities must stay between 0 and 1
assert np.all(probabilities >= 0)
assert np.all(probabilities <= 1)

# A logit of zero must give probability 0.5
assert probabilities[3] == 0.5

# Extreme negative and positive values
assert probabilities[0] == 0.0
assert probabilities[-1] == 1.0

print("\nStable sigmoid tests: PASSED")

# Extreme scores and their correct labels
extreme_logits = np.array([-1000.0, 1000.0])

correct_labels = np.array([0, 1])

good_loss = LogisticClassifier.binary_loss(extreme_logits, correct_labels)

# Deliberately incorrect labels
wrong_labels = np.array([1,0])

bad_loss = LogisticClassifier.binary_loss(extreme_logits,wrong_labels)

print("\nGood prediction loss:", good_loss)
print("Bad prediction loss:", bad_loss)

assert np.isfinite(good_loss)
assert np.isfinite(bad_loss)

assert good_loss < bad_loss

print("Stable loss tests: PASSED")

def numerical_gradient(X, y, parameters, h=1e-5):

    gradient = np.zeros_like(parameters)

    for j in range(len(parameters)):

        parameters_plus = parameters.copy()
        parameters_minus = parameters.copy()

        parameters_plus[j] += h
        parameters_minus[j] -= h

        loss_plus, _ = LogisticClassifier._loss_and_gradient(X, y, parameters_plus)

        loss_minus, _ = LogisticClassifier._loss_and_gradient(X, y, parameters_minus)

        gradient[j] = (loss_plus - loss_minus) / (2 * h)

    return gradient

print("\nGRADIENT CHECK")

rng = np.random.default_rng(42)

# 30 observations, 3 numerical features
X_check = rng.normal(size=(30, 3))

# Binary target labels
y_check = rng.integers(0, 2, size=30)

# Four parameters: one intercept + three weights
parameter_vectors = [
    np.zeros(4),
    np.array([0.2, -0.3, 0.5, 0.1]),
    np.array([-1.0, 0.8, -0.6, 1.2]),
    np.array([2.0, -1.0, 3.0, -2.0]),
    np.array([10.0, -8.0, 12.0, -15.0])
]

for i, parameters in enumerate(parameter_vectors, start=1):

    # Analytical gradient
    loss, analytical = LogisticClassifier._loss_and_gradient(X_check, y_check, parameters)

    # Numerical gradient
    numerical = numerical_gradient(X_check, y_check, parameters)

    # Find the largest disagreement
    max_difference = np.max(np.abs(analytical - numerical))

    print(
        f"Vector {i}: "
        f"loss={loss:.6f}, "
        f"max gradient difference={max_difference:.12e}"
    )

    assert max_difference < 1e-5

print("All five gradient checks: PASSED")

print("\nWEIGHTED + REGULARIZED GRADIENT CHECK")

parameters = np.array([
    0.5, -0.3, 0.8, -0.2
])

sample_weights = np.where(
    y_check == 1,
    2.0,
    0.5
)

sample_weights = (
    sample_weights / sample_weights.mean()
)

l2_strength = 0.1
h = 1e-5

loss, analytical = LogisticClassifier._loss_and_gradient(
    X_check,
    y_check,
    parameters,
    l2=l2_strength,
    sample_weights=sample_weights
)

numerical = np.zeros_like(parameters)

for j in range(len(parameters)):

    plus = parameters.copy()
    minus = parameters.copy()

    plus[j] += h
    minus[j] -= h

    loss_plus, _ = LogisticClassifier._loss_and_gradient(
        X_check,
        y_check,
        plus,
        l2=l2_strength,
        sample_weights=sample_weights
    )

    loss_minus, _ = LogisticClassifier._loss_and_gradient(
        X_check,
        y_check,
        minus,
        l2=l2_strength,
        sample_weights=sample_weights
    )

    numerical[j] = (
        loss_plus - loss_minus
    ) / (2 * h)

max_difference = np.max(
    np.abs(analytical - numerical)
)

print("Weighted gradient difference:", max_difference)

assert max_difference < 1e-5

print("Weighted + L2 gradient check: PASSED")

print("\nFIRST CLASSIFIER TRAINING")

rng = np.random.default_rng(42)

# Create 500 observations with two features
X_train_demo = rng.normal(size=(500, 2))

# Define the underlying relationship
true_logits = (
    -0.5
    + 1.5 * X_train_demo[:, 0]
    - 1.0 * X_train_demo[:, 1]
)

# Convert the logits into probabilities
true_probabilities = LogisticClassifier.sigmoid(
    true_logits
)

# Draw binary labels using those probabilities
y_train_demo = rng.binomial(
    1,
    true_probabilities
)

# Create our classifier
model = LogisticClassifier(
    lr=0.1,
    steps=10000,
    l2=0.01,
    tol=1e-8
)

# Train it
model.fit(X_train_demo, y_train_demo)

print("Learned intercept:", model.intercept_)
print("Learned weights:", model.coef_)
print("Training accuracy:", model.score(
    X_train_demo,
    y_train_demo
))
print("Iterations used:", model.n_iter_)
print("Converged at:", model.converged_at)
print("Initial loss:", model.loss_history_[0])
print("Final loss:", model.loss_history_[-1])

assert np.isfinite(model.w).all()
assert np.isfinite(model.loss_history_).all()
assert model.loss_history_[-1] <= model.loss_history_[0]

print("Classifier training tests: PASSED")

print("\nINPUT VALIDATION TESTS")


def expect_error(X_bad, y_bad, expected_text):

    try:
        LogisticClassifier().fit(X_bad, y_bad)

    except ValueError as error:
        assert expected_text.lower() in str(error).lower()
        return

    raise AssertionError(
        f"Expected a ValueError containing: {expected_text}"
    )


# Invalid class label
expect_error(
    [[1], [2], [3]],
    [0, 1, 2],
    "Labels must contain only 0 and 1"
)

# Different lengths
expect_error(
    [[1], [2], [3]],
    [0, 1],
    "matching lengths"
)

# Missing numerical feature
expect_error(
    [[1], [np.nan], [3]],
    [0, 1, 0],
    "NaN"
)

# Missing label
expect_error(
    [[1], [2], [3]],
    [0, np.nan, 1],
    "Labels"
)

# Only one class
expect_error(
    [[1], [2], [3]],
    [0, 0, 0],
    "both class"
)

print("Input validation tests: PASSED")