import numpy as np


class LogisticClassifier:

    def __init__(
        self,
        lr=0.1,
        steps=10000,
        l2=0.0,
        class_weight=None,
        tol=1e-9
    ):
        self.lr = lr
        self.steps = steps
        self.l2 = l2
        self.class_weight = class_weight
        self.tol = tol

    @staticmethod
    def sigmoid(z):
        z = np.asarray(z, dtype=float)
        result = np.empty_like(z)

        positive = z >= 0
        negative = ~positive

        result[positive] = (
            1 / (1 + np.exp(-z[positive]))
        )

        exp_z = np.exp(z[negative])
        result[negative] = exp_z / (1 + exp_z)

        return result

    @staticmethod
    def binary_loss(logits, y):
        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y, dtype=float)

        losses = np.logaddexp(0, logits) - y * logits

        return np.mean(losses)

    @staticmethod
    def _loss_and_gradient(
        X,
        y,
        parameters,
        l2=0.0,
        sample_weights=None
    ):
        n = len(y)

        intercept = parameters[0]
        weights = parameters[1:]

        logits = X @ weights + intercept
        probabilities = LogisticClassifier.sigmoid(logits)

        if sample_weights is None:
            sample_weights = np.ones(n)

        individual_losses = (
            np.logaddexp(0, logits) - y * logits
        )

        data_loss = np.mean(
            sample_weights * individual_losses
        )

        penalty = 0.5 * l2 * np.sum(weights ** 2)

        loss = data_loss + penalty

        errors = sample_weights * (probabilities - y)

        intercept_gradient = np.mean(errors)

        weight_gradient = (
            X.T @ errors / n + l2 * weights
        )

        gradient = np.concatenate([
            [intercept_gradient],
            weight_gradient
        ])

        return loss, gradient

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError(
                "X must be a non-empty 1-D or 2-D array."
            )

        if X.shape[1] == 0:
            raise ValueError("X must contain features.")

        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values."
            )

        return X

    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")

        if len(X) != len(y):
            raise ValueError(
                "X and y must have matching lengths."
            )

        if not np.isin(y, [0, 1]).all():
            raise ValueError(
                "Labels must contain only 0 and 1."
            )

        if np.unique(y).size != 2:
            raise ValueError(
                "Training requires both class 0 and class 1."
            )

        if not np.isfinite(y).all():
            raise ValueError("y contains invalid values.")

        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite.")

        if not isinstance(self.steps, (int, np.integer)):
            raise ValueError("steps must be an integer.")

        if self.steps <= 0:
            raise ValueError("steps must be positive.")

        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError(
                "l2 must be non-negative and finite."
            )

        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be positive and finite.")

        if self.class_weight not in (None, "balanced"):
            raise ValueError(
                "class_weight must be None or 'balanced'."
            )

        y = y.astype(float)
        n_samples, n_features = X.shape

        if self.class_weight == "balanced":
            counts = np.bincount(
                y.astype(int),
                minlength=2
            )

            if np.any(counts == 0):
                raise ValueError(
                    "Balanced weighting requires both classes."
                )

            class_weights = n_samples / (2 * counts)

            sample_weights = class_weights[
                y.astype(int)
            ]

            sample_weights = (
                sample_weights / sample_weights.mean()
            )

        else:
            sample_weights = np.ones(n_samples)

        parameters = np.zeros(n_features + 1)

        self.loss_history_ = []
        self.converged_at = None

        for step in range(1, self.steps + 1):

            loss, gradient = self._loss_and_gradient(
                X,
                y,
                parameters,
                l2=self.l2,
                sample_weights=sample_weights
            )

            self.loss_history_.append(loss)

            new_parameters = (
                parameters - self.lr * gradient
            )

            max_change = np.max(
                np.abs(new_parameters - parameters)
            )

            parameters = new_parameters

            if max_change < self.tol:
                self.converged_at = step
                break

        final_loss, _ = self._loss_and_gradient(
            X,
            y,
            parameters,
            l2=self.l2,
            sample_weights=sample_weights
        )

        self.loss_history_.append(final_loss)

        self.w = parameters.copy()
        self.intercept_ = parameters[0]
        self.coef_ = parameters[1:]
        self.n_iter_ = step

        return self

    def predict_proba(self, X):
        if not hasattr(self, "w"):
            raise ValueError(
                "Fit the model before predicting."
            )

        X = self._validate_X(X)

        if X.shape[1] != len(self.coef_):
            raise ValueError(
                "Number of features does not match training."
            )

        logits = X @ self.coef_ + self.intercept_
        return self.sigmoid(logits)

    def predict(self, X, threshold=0.5):
        if not np.isfinite(threshold) or not (
            0 <= threshold <= 1
        ):
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        probabilities = self.predict_proba(X)

        return (
            probabilities >= threshold
        ).astype(int)

    def score(self, X, y):
        y = np.asarray(y)
        predictions = self.predict(X)

        if y.ndim != 1 or len(y) != len(predictions):
            raise ValueError(
                "y must match the number of predictions."
            )

        if not np.isin(y, [0, 1]).all():
            raise ValueError(
                "Labels must contain only 0 and 1."
            )

        return np.mean(predictions == y)