
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import warnings

class MyLinearRegression:

    def __init__(
        self,
        alpha=0.0,
        penalty="l2",
        fit_intercept=True
    ):
        self.alpha = alpha
        self.penalty = penalty
        self.fit_intercept = fit_intercept

    @staticmethod
    def _soft_threshold(z, gamma):
        return np.sign(z) * max(abs(z) - gamma, 0.0)
    
    def fit(self, X, y):

        # Convert input data into NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Allow a single feature as a 1-D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Check input dimensions
        if X.ndim != 2:
            raise ValueError("X must be a 1-D or 2-D array.")

        if y.ndim != 1:
            raise ValueError("y must be a 1-D array.")

        # Check for empty inputs
        if X.size == 0 or y.size == 0:
            raise ValueError("Input data cannot be empty.")

        # Check that X and y have matching rows
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        # Check for NaN or infinite values
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinite values.")

        # OLS only for this first version
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")

        if self.penalty not in ['l1', 'l2']:
            raise ValueError("penalty must be l1 or l2.")

        # Centre X and y when fitting an intercept
        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()

            X_centered = X - X_mean
            y_centered = y - y_mean

        else:
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0

            X_centered = X
            y_centered = y

        if self.alpha == 0:

            coefficients = np.linalg.lstsq(
                X_centered,
                y_centered,
                rcond=None
            )[0]

        elif self.penalty == "l2":

            n_features = X_centered.shape[1]

            identity = np.eye(n_features)

            A = (
                X_centered.T @ X_centered
                + self.alpha * identity
            )

            b = X_centered.T @ y_centered

            coefficients = np.linalg.solve(A, b)

        elif self.penalty == "l1":

            coefficients = self._fit_lasso(
                X_centered,
                y_centered
            )

        self.coef_ = coefficients

        # Recover the intercept
        self.intercept_ = (
            y_mean - X_mean @ coefficients
        )

        return self

    def _fit_lasso(self, X, y):

        n_samples, n_features = X.shape

        weights = np.zeros(n_features)

        max_iter = 100000
        tol = 1e-12

        column_norms = np.sum(X ** 2, axis=0) / n_samples

        for iteration in range(max_iter):

            old_weights = weights.copy()

            for j in range(n_features):

                if column_norms[j] == 0:
                    weights[j] = 0.0
                    continue

                residual = y - X @ weights

                partial_residual = (
                    residual + X[:, j] * weights[j]
                )

                rho = (
                    X[:, j] @ partial_residual
                ) / n_samples

                weights[j] = (
                    self._soft_threshold(rho, self.alpha)
                    / column_norms[j]
                )

            max_change = np.max(
                np.abs(weights - old_weights)
            )

            if max_change < tol:
                break

        self.n_iter_ = iteration + 1

        return weights

    def predict(self, X):

        if not hasattr(self, "coef_"):
            raise ValueError("Fit the model before predicting.")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1-D or 2-D array.")

        if X.shape[1] != len(self.coef_):
            raise ValueError(
                "Number of features does not match training data."
            )

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        return X @ self.coef_ + self.intercept_

    def score(self, X, y):

        y = np.asarray(y, dtype=float)
        predictions = self.predict(X)

        if y.ndim != 1:
            raise ValueError("y must be a 1-D array.")

        if len(y) != len(predictions):
            raise ValueError("X and y lengths do not match.")

        if not np.isfinite(y).all():
            raise ValueError("y contains invalid values.")

        if len(y) < 2:
            raise ValueError("R2 requires at least two observations.")

        ss_res = np.sum((y - predictions) ** 2)
        ss_total = np.sum((y - y.mean()) ** 2)

        if ss_total == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - ss_res / ss_total

    def residual_report(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = self.predict(X)
        residuals = y - predictions

        if len(residuals) < 3:
            raise ValueError(
                "At least three observations are needed for diagnostics."
            )

        # Linearity: residual association with squared fitted values
        correlation = np.corrcoef(
            predictions ** 2,
            residuals
        )[0, 1]

        # Independence: Durbin-Watson statistic
        dw_stat = durbin_watson(residuals)

        # Homoscedasticity: Breusch-Pagan test
        X_bp = np.column_stack([
            np.ones(len(predictions)),
            predictions
        ])

        bp_stat, bp_pvalue, _, _ = het_breuschpagan(
            residuals,
            X_bp
        )

        # Normality: Jarque-Bera test
        jb_result = stats.jarque_bera(residuals)

        report = {
            "linearity": {
                "squared_fitted_residual_correlation": float(correlation)
            },
            "independence": {
                "durbin_watson": float(dw_stat)
            },
            "homoscedasticity": {
                "breusch_pagan_statistic": float(bp_stat),
                "p_value": float(bp_pvalue)
            },
            "normality": {
                "jarque_bera_statistic": float(jb_result.statistic),
                "p_value": float(jb_result.pvalue)
            }
        }

        return report