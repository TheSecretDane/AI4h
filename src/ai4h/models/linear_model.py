from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Tuple
import numpy as np


def test():
    project_root = Path(__file__).resolve().parents[3]
    relpath = Path(__file__).resolve().relative_to(project_root)
    print(f"Hello from `{relpath}`!")

def expit_own(X, beta):
    lin_com = X.T @ beta
    return 1 / (1 + np.exp(-lin_com))

def expit(z: np.ndarray) -> np.ndarray:
    """
    Stable elementwise sigmoid σ(z) = 1/(1+e^{-z}).
    Accepts scalar or array-like input, returns ndarray with same shape.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)

    # For non-negative z use standard form, for negative use alternative to avoid overflow
    pos = z >= 0
    if np.any(pos):
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    if np.any(~pos):
        ez = np.exp(z[~pos])        # safe because z[~pos] < 0
        out[~pos] = ez / (1.0 + ez)
    return out

def generate_logreg_data(
    n: int,
    beta: np.ndarray,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate logistic regression data.

    Returns:
      X : ndarray shape (n, p+1)  -- first column ones (intercept)
      y : ndarray shape (n,)       -- {0,1} outcomes

    beta is length p+1 (intercept first).
    """
    beta = np.asarray(beta, dtype=float).ravel()
    p = beta.size - 1
    rng = np.random.default_rng(seed)

    X = np.empty((n, p + 1), dtype=float)
    X[:, 0] = 1.0
    if p > 0:
        X[:, 1:] = rng.uniform(-1.0, 1.0, size=(n, p))

    # linear predictor and probabilities
    eta = X @ beta                # shape (n,)
    probs = expit(eta)            # elementwise sigmoid
    y = rng.binomial(1, probs, size=n).astype(np.int64)
    return X, y

def irls(X, y, tol=1e-8, max_iter=100):
    """
    Simple IRLS (Newton-Raphson) for logistic regression.
    X : (n, k) design matrix (include intercept column if desired)
    y : (n,) or (n,1) with values 0 or 1
    Returns: (beta, n_iter, converged) where beta is 1D array length k
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1, 1).astype(float)
    n, k = X.shape

    # start at zero
    beta = np.zeros((k, 1), dtype=float)
    eps_weight = 1e-12   # avoid dividing by exact zero
    ridge = 1e-8         # tiny regularizer if matrix is singular

    for it in range(1, max_iter + 1):
        eta = X @ beta                    # (n,1)
        p = expit(eta)                    # (n,1)
        W_vec = (p * (1.0 - p)).reshape(-1)   # (n,)

        # avoid zeros in W
        W_safe = np.maximum(W_vec, eps_weight)

        # adjusted response (z) - equation (13)
        z = eta + (y - p) / W_safe.reshape(-1, 1)   # (n,1)

        # build weighted normal equations
        # X^T W X  and  X^T W z
        WX = X * W_safe.reshape(-1, 1)              # (n,k)
        XT_W_X = X.T @ WX                           # (k,k)
        XT_W_X_inv = np.linalg.pinv(XT_W_X)  # for debugging only
        XT_W_z = X.T @ (W_safe.reshape(-1,1) * z)   # (k,1)

        # solve (use tiny ridge if singular)
        try:
            beta_next = np.linalg.solve(XT_W_X, XT_W_z)
        except np.linalg.LinAlgError:
            beta_next = np.linalg.solve(XT_W_X + ridge * np.eye(k), XT_W_z)

        # convergence check (max absolute change)
        if np.max(np.abs(beta_next - beta)) <= tol:
            beta = beta_next
            return beta.ravel(), it, True

        beta = beta_next

    return beta.ravel(), max_iter, False

class LogisticRegression:
    def __init__(self):
        self.coef_ = None
        self.n_iter_ = None
        self.converged_ = None

    # def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]):
    #     """Estimate the Logistic Regression model using IRLS"""
    #     beta, n_iter, converged = irls(X, y)
    #     self.coef_ = result["estimates"]
    #     self.n_iter_ = result["iterations"]
    #     self.converged_ = result["converged"]

    def fit(self, X, y, **irls_kwargs):
        """
        Estimate the Logistic Regression model using IRLS.
        If irls returns a dict, we accept that; if it returns a tuple, we accept that too.
        """
        res = irls(X, y, **irls_kwargs)
        # support both return styles: dict or tuple
        if isinstance(res, dict):
            self.coef_ = res.get("estimates")
            self.n_iter_ = int(res.get("iterations", 0))
            self.converged_ = bool(res.get("converged", False))
        else:
            # assume tuple (beta, n_iter, converged)
            beta, n_iter, converged = res
            self.coef_ = np.asarray(beta).ravel()
            self.n_iter_ = int(n_iter)
            self.converged_ = bool(converged)
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute predicted probabilities."""
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        return expit(np.asarray(X, dtype=float) @ np.asarray(self.coef_))

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict binary outcomes based on a 0.5 threshold."""
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        return (self.predict_proba(X) >= 0.5).astype(int)
