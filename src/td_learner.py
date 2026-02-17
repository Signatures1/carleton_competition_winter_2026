"""
Bridging Supervised Learning and Reinforcement Learning
via Markov Reward Process (MRP) Reformulation
=======================================================

Key insight
-----------
Standard supervised learning (SL) minimises E[(V(x) - y)²] with no sense
of ordering between training examples.  When the target noise is
*autocorrelated* — a common real-world problem — OLS is inefficient: the
Gauss-Markov theorem guarantees BLUE only under i.i.d. errors.

We reframe the SL dataset as a *Markov Reward Process*:

    • State     sᵢ  =  feature vector φ(xᵢ)
    • Reward    rᵢ  =  target yᵢ
    • Transition     sᵢ  →  s_{i+1}  (next example in the ordered dataset)
    • Goal           learn  V  satisfying the Bellman equation:
                     V(sᵢ) = rᵢ + γ · V(s_{i+1})

The LSTD(γ) fixed-point solves the linear system  A θ = b  where:

    A  =  Φᵀ (Φ − γ Φ')          ← Φ' is Φ row-shifted by one position
    b  =  Φᵀ r

Setting γ = 0 recovers ordinary least squares (OLS) exactly.
Setting γ = ρ (the AR(1) noise coefficient) is asymptotically equivalent
to the *Prais-Winsten* Generalised Least Squares (GLS) transformation:

    ỹᵢ = yᵢ − ρ yᵢ₋₁    and    x̃ᵢ = xᵢ − ρ xᵢ₋₁

which whitens AR(1) noise and achieves the BLUE / minimum-variance estimate.

Inverse-link generalisation
----------------------------
For non-Gaussian targets the identity link is replaced by  g⁻¹:

    V(xᵢ) = g⁻¹(θᵀ φ(xᵢ))

Support:
    IDENTITY   continuous regression   g⁻¹(η) = η
    LOGIT      binary classification   g⁻¹(η) = σ(η) = 1 / (1 + exp(−η))
    LOG        count / Poisson         g⁻¹(η) = exp(η)

The TD version of Iteratively Reweighted Least Squares (TD-IRLS) applies
the standard IRLS linearisation but replaces  Φᵀ W Φ  with  Φᵀ W (Φ − γ Φ'),
inheriting all TD benefits while handling non-identity links.

References
----------
Bradtke & Barto (1996)  "Linear Least-Squares Algorithms for Temporal
    Difference Learning"  Machine Learning 22(1-3).
Prais & Winsten (1954)  "Trend Estimators and Serial Correlation"
    Cowles Commission Discussion Paper.
"""

from __future__ import annotations

import warnings
import numpy as np

__all__ = ["MRPDataset", "LSTDLearner", "GammaSearchCV"]


# ============================================================================
# Section 1 — Utilities
# ============================================================================

def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X."""
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, np.asarray(X, dtype=float)])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: σ(z) = 1 / (1 + exp(−z))."""
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _link_fn(link: str, eta: np.ndarray):
    """
    Apply the inverse link and return (mu, W_diag).

    W_diag[i] is the IRLS weight at observation i (= variance function
    evaluated at μᵢ for canonical exponential-family distributions).

    Returns
    -------
    mu     : (n,) array of fitted means
    W_diag : (n,) array of IRLS weights
    """
    eta = np.asarray(eta, dtype=float)
    if link == "identity":
        mu = eta.copy()
        W_diag = np.ones_like(mu)
    elif link == "logit":
        mu = _sigmoid(eta)
        mu = np.clip(mu, 1e-8, 1.0 - 1e-8)
        W_diag = mu * (1.0 - mu)
    elif link == "log":
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        W_diag = mu
    else:
        raise ValueError(f"Unknown link '{link}'. Choose: 'identity', 'logit', 'log'.")
    W_diag = np.maximum(W_diag, 1e-10)   # guard against division-by-zero
    return mu, W_diag


def _neg_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    ss_res = np.sum((y_true - np.asarray(y_pred, dtype=float)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-15))


def _neg_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 1e-10, 1.0 - 1e-10)
    return float(np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def _score(link: str, scoring: str, y: np.ndarray, y_hat: np.ndarray) -> float:
    """Return a scalar score (higher = better)."""
    if scoring == "r2":
        return _r2(y, y_hat)
    if scoring == "neg_mse":
        return _neg_mse(y, y_hat)
    if scoring == "neg_log_loss":
        return _neg_log_loss(y, y_hat)
    raise ValueError(f"Unknown scoring '{scoring}'.")


# ============================================================================
# Section 2 — MRPDataset
# ============================================================================

class MRPDataset:
    """
    Wraps a supervised (X, y) dataset as a one-step Markov Reward Process.

    The transition model is a fixed sequential shift: state sᵢ transitions
    to state s_{i+1}, with wrap-around at the end (sₙ → s₀).  Every example
    acts as both a "current" state and a "next" state for its predecessor,
    making the chain ergodic under a uniform stationary distribution.

    Parameters
    ----------
    X : array-like, shape (n, d)
    y : array-like, shape (n,)
    add_intercept : bool, default True
        If True a column of ones is prepended to X before constructing Φ.
    """

    def __init__(self, X, y, add_intercept: bool = True):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim > 2:
            raise ValueError("X must be 1-D or 2-D.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y = np.asarray(y, dtype=float).ravel()
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if len(y) < 2:
            raise ValueError("Dataset must contain at least 2 examples.")

        self.Phi: np.ndarray = _add_intercept(X) if add_intercept else X.copy()
        self.r:   np.ndarray = y
        self.n,   self.p     = self.Phi.shape

        # Sequential shift with wrap-around: Φ'[i] = Φ[(i+1) % n]
        self.Phi_prime: np.ndarray = np.roll(self.Phi, shift=-1, axis=0)

    # ------------------------------------------------------------------
    def get_A(self, gamma: float) -> np.ndarray:
        """Return  Φᵀ(Φ − γ Φ')  — the unweighted LSTD matrix."""
        return self.Phi.T @ (self.Phi - gamma * self.Phi_prime)

    def get_b(self) -> np.ndarray:
        """Return  Φᵀ r  — the unweighted LSTD right-hand side."""
        return self.Phi.T @ self.r

    def get_weighted_A(self, gamma: float, W_diag: np.ndarray) -> np.ndarray:
        """Return  Φᵀ diag(W) (Φ − γ Φ')  — weighted LSTD matrix for IRLS."""
        W = W_diag[:, None]   # broadcast-friendly column vector
        return (self.Phi * W).T @ (self.Phi - gamma * self.Phi_prime)

    def get_weighted_b(self, W_diag: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return  Φᵀ diag(W) z  — weighted LSTD RHS for IRLS."""
        return (self.Phi * W_diag[:, None]).T @ z


# ============================================================================
# Section 3 — LSTDLearner
# ============================================================================

class LSTDLearner:
    """
    Least-Squares Temporal Difference (LSTD) Supervised Learner.

    Solves the LSTD(γ) fixed-point equation  A θ = b  where

        A  =  Φᵀ (Φ − γ Φ')
        b  =  Φᵀ y          (or the IRLS-adjusted quantities for GLM links)

    For γ = 0 this is exactly OLS.
    For γ > 0 the estimate can strictly outperform OLS when successive
    training examples have correlated errors (e.g. AR(1) time series).

    Parameters
    ----------
    gamma : float in [0, 1)
        TD discount.  0 = pure OLS; set to the AR(1) coefficient for
        optimal GLS-equivalent performance.
    link : {"identity", "logit", "log"}
        Inverse link function.
    add_intercept : bool
        Whether to add a bias column.
    lambda_reg : float
        Ridge penalty added to A before solving (numerical stability).
    max_irls_iter : int
        Maximum IRLS iterations for non-identity links.
    irls_tol : float
        Convergence criterion: max absolute change in η between iterations.
    verbose : bool
        Print convergence info.
    """

    def __init__(
        self,
        gamma: float = 0.0,
        link: str = "identity",
        add_intercept: bool = True,
        lambda_reg: float = 1e-10,
        max_irls_iter: int = 50,
        irls_tol: float = 1e-7,
        verbose: bool = False,
    ):
        if not (0.0 <= gamma < 1.0):
            raise ValueError("gamma must be in [0, 1).")
        if link not in ("identity", "logit", "log"):
            raise ValueError("link must be 'identity', 'logit', or 'log'.")

        self.gamma         = gamma
        self.link          = link
        self.add_intercept = add_intercept
        self.lambda_reg    = lambda_reg
        self.max_irls_iter = max_irls_iter
        self.irls_tol      = irls_tol
        self.verbose       = verbose

        self.theta_:    np.ndarray | None = None
        self.dataset_:  MRPDataset | None = None
        self.n_iter_:   int  = 0
        self.converged_: bool = False

    # ------------------------------------------------------------------
    def fit(self, X, y) -> LSTDLearner:
        """
        Fit the LSTD model.

        Parameters
        ----------
        X : array-like, shape (n, d)
        y : array-like, shape (n,)

        Returns
        -------
        self
        """
        self.dataset_ = MRPDataset(X, y, add_intercept=self.add_intercept)
        Phi, Phi_prime, r = (
            self.dataset_.Phi,
            self.dataset_.Phi_prime,
            self.dataset_.r,
        )
        p = self.dataset_.p
        ridge = self.lambda_reg * np.eye(p)

        if self.link == "identity":
            # ---- Linear LSTD: solve A θ = b directly ----
            A = self.dataset_.get_A(self.gamma) + ridge
            b = self.dataset_.get_b()
            self.theta_, *_ = np.linalg.lstsq(A, b, rcond=None)
            self.n_iter_    = 1
            self.converged_ = True

        else:
            # ---- TD-IRLS for non-identity links ----
            # When gamma > 0 the A matrix Phi.T @ W @ (Phi - gamma*Phi') is
            # non-symmetric and can be indefinite, making vanilla IRLS diverge.
            # We use step-size damping (backtracking on eta norm) to stabilise.
            #
            # Warm-start: solve OLS to get an initial eta
            A0 = Phi.T @ Phi + ridge
            b0 = Phi.T @ r
            theta, *_ = np.linalg.lstsq(A0, b0, rcond=None)
            eta = Phi @ theta

            # Stronger ridge for non-identity links (helps with indefinite A)
            ridge_irls = max(self.lambda_reg, 1e-4) * np.eye(p)

            self.converged_ = False
            prev_norm = np.linalg.norm(eta)

            for i in range(self.max_irls_iter):
                eta_old = eta.copy()
                mu, W_diag = _link_fn(self.link, eta)

                # Working (adjusted) response
                z = eta + (r - mu) / W_diag

                # Weighted LSTD system
                A_w = self.dataset_.get_weighted_A(self.gamma, W_diag) + ridge_irls
                b_w = self.dataset_.get_weighted_b(W_diag, z)
                theta_new, *_ = np.linalg.lstsq(A_w, b_w, rcond=None)
                eta_proposed = Phi @ theta_new

                # Backtracking step-size damping: if eta norm explodes, shrink step
                alpha = 1.0
                for _ in range(8):
                    eta_candidate = (1.0 - alpha) * eta_old + alpha * eta_proposed
                    if np.linalg.norm(eta_candidate) <= 4.0 * max(prev_norm, 1.0):
                        break
                    alpha *= 0.5
                else:
                    eta_candidate = eta_old   # full rejection; stay put

                theta = (1.0 - alpha) * theta + alpha * theta_new
                eta   = eta_candidate
                prev_norm = np.linalg.norm(eta)

                delta = np.max(np.abs(eta - eta_old))
                if self.verbose:
                    print(f"  IRLS iter {i+1:3d}  |  delta_eta = {delta:.3e}  alpha = {alpha:.3f}")

                if delta < self.irls_tol:
                    self.converged_ = True
                    self.n_iter_    = i + 1
                    break
            else:
                self.n_iter_ = self.max_irls_iter
                if delta > 1.0:   # only warn if truly diverged, not just slow
                    warnings.warn(
                        f"TD-IRLS did not converge after {self.max_irls_iter} iterations "
                        f"(final delta_eta = {delta:.3e}). Returning last iterate.",
                        RuntimeWarning,
                    )
                else:
                    self.converged_ = True   # slow but numerically stable

            self.theta_ = theta

        return self

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """Return fitted values (in mean / response space, not link space)."""
        if self.theta_ is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.add_intercept:
            X = _add_intercept(X)
        eta = X @ self.theta_
        mu, _ = _link_fn(self.link, eta)
        return mu

    # ------------------------------------------------------------------
    def score(self, X, y) -> float:
        """
        Compute a goodness-of-fit score (higher = better).

        Returns
        -------
        float
            R²  for identity link,
            negative log-loss  for logit,
            negative mean MSE  for log.
        """
        y_hat = self.predict(X)
        if self.link == "identity":
            return _r2(y, y_hat)
        if self.link == "logit":
            return _neg_log_loss(y, y_hat)
        # log link: use neg-MSE as a reasonable proxy
        return _neg_mse(y, y_hat)


# ============================================================================
# Section 4 — GammaSearchCV
# ============================================================================

class GammaSearchCV:
    """
    Cross-validated search over the TD discount γ.

    Finds the γ that maximises a held-out score metric on k folds,
    then re-fits an ``LSTDLearner`` on the full training data.

    Parameters
    ----------
    gamma_grid : array-like or None
        Candidate γ values.  Defaults to ``np.linspace(0, 0.95, 40)``.
    link : str
        Passed to ``LSTDLearner``.
    cv : int
        Number of cross-validation folds.
    scoring : {"neg_mse", "r2", "neg_log_loss"}
        Score function (higher = better in all cases).
    add_intercept : bool
    lambda_reg : float
    verbose : bool
    """

    def __init__(
        self,
        gamma_grid=None,
        link: str = "identity",
        cv: int = 5,
        scoring: str = "neg_mse",
        add_intercept: bool = True,
        lambda_reg: float = 1e-10,
        verbose: bool = False,
    ):
        self.gamma_grid    = (
            np.linspace(0.0, 0.95, 40)
            if gamma_grid is None
            else np.asarray(gamma_grid, dtype=float)
        )
        self.link          = link
        self.cv            = cv
        self.scoring       = scoring
        self.add_intercept = add_intercept
        self.lambda_reg    = lambda_reg
        self.verbose       = verbose

        self.best_gamma_:     float | None       = None
        self.best_score_:     float | None       = None
        self.cv_results_:     dict | None        = None
        self.best_estimator_: LSTDLearner | None = None

    # ------------------------------------------------------------------
    def fit(self, X, y) -> GammaSearchCV:
        """Run cross-validated grid search and refit the best model."""
        from sklearn.model_selection import KFold

        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        # Safety: cap cv at number of samples
        cv = min(self.cv, n)
        kf = KFold(n_splits=cv, shuffle=False)

        # Filter γ = 1 values
        gamma_grid = self.gamma_grid[self.gamma_grid < 1.0]
        if len(gamma_grid) < len(self.gamma_grid):
            warnings.warn("gamma_grid values >= 1 have been removed.", UserWarning)

        mean_scores = np.empty(len(gamma_grid))
        std_scores  = np.empty(len(gamma_grid))

        for g_idx, gamma in enumerate(gamma_grid):
            fold_scores = []
            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                m = LSTDLearner(
                    gamma=float(gamma),
                    link=self.link,
                    add_intercept=self.add_intercept,
                    lambda_reg=self.lambda_reg,
                )
                m.fit(X_tr, y_tr)
                y_hat = m.predict(X_val)
                fold_scores.append(_score(self.link, self.scoring, y_val, y_hat))

            mean_scores[g_idx] = np.mean(fold_scores)
            std_scores[g_idx]  = np.std(fold_scores)
            if self.verbose:
                print(f"  γ = {gamma:.3f}  │  mean {self.scoring} = {mean_scores[g_idx]:+.5f}"
                      f"  ± {std_scores[g_idx]:.5f}")

        best_idx          = int(np.argmax(mean_scores))
        self.best_gamma_  = float(gamma_grid[best_idx])
        self.best_score_  = float(mean_scores[best_idx])
        self.cv_results_  = {
            "gamma":      gamma_grid,
            "mean_score": mean_scores,
            "std_score":  std_scores,
        }

        # Refit on full data with best γ
        self.best_estimator_ = LSTDLearner(
            gamma=self.best_gamma_,
            link=self.link,
            add_intercept=self.add_intercept,
            lambda_reg=self.lambda_reg,
        ).fit(X, y)

        return self

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """Delegate to the best fitted estimator."""
        if self.best_estimator_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.best_estimator_.predict(X)

    # ------------------------------------------------------------------
    def plot_gamma_curve(self, ax=None):
        """
        Plot mean ± std CV score vs γ, marking the best γ.

        Requires matplotlib.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if self.cv_results_ is None:
            raise RuntimeError("Call fit() before plot_gamma_curve().")

        r = self.cv_results_
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        ax.plot(r["gamma"], r["mean_score"], color="steelblue", linewidth=2,
                label=f"CV {self.scoring}")
        ax.fill_between(
            r["gamma"],
            r["mean_score"] - r["std_score"],
            r["mean_score"] + r["std_score"],
            alpha=0.2, color="steelblue", label="±1 SD",
        )
        ax.axvline(self.best_gamma_, color="crimson", linestyle="--",
                   label=f"best γ = {self.best_gamma_:.3f}")
        ax.set_xlabel("γ  (TD discount factor)")
        ax.set_ylabel(self.scoring)
        ax.set_title("LSTD(γ) — cross-validated score vs. discount")
        ax.legend()
        return ax


# ============================================================================
# Section 5 — Synthetic data generators & comparison utilities
# ============================================================================

def generate_mrp_data(
    n: int = 300,
    d: int = 5,
    rho_x: float = 0.7,
    gamma: float = 0.5,
    sigma: float = 0.5,
    seed: int = 42,
):
    """
    Generate data from a linear Markov Reward Process.

    Model
    -----
        x_{t+1} = ρ_x · x_t + √(1−ρ_x²) · η_t    [AR(1) Markov states]
        V*(x)   = θ*ᵀ x                              [linear true value fn]
        r_t = V*(x_t) − γ V*(x_{t+1}) + ε_t         [noisy Bellman reward]
            = (1 − γ ρ_x) · θ*ᵀ x_t + noise         [effective linear form]

    Why this matters
    ----------------
    OLS regresses x_t on r_t and recovers:
        θ_OLS  ≈  θ* · (1 − γ ρ_x)        ← BIASED by factor (1−γρ_x)

    LSTD(γ) solves  Aθ = b  with  A = Φᵀ(Φ − γΦ'), recovering:
        θ_LSTD ≈  θ*                         ← UNBIASED

    The bias of OLS is  Δ = θ* · γ ρ_x.
    For γ=0.9, ρ_x=0.9: Δ = 0.81 θ*  — OLS underestimates by 81 %!

    Returns
    -------
    X          : (n, d)  feature matrix (sequential AR(1) states)
    r          : (n,)    immediate rewards (noisy Bellman returns)
    theta_true : (d,)    true value-function coefficients θ*
    """
    rng = np.random.default_rng(seed)
    theta_true = rng.standard_normal(d)

    # Stationary AR(1) states
    X = np.zeros((n, d))
    X[0] = rng.standard_normal(d)
    for i in range(1, n):
        X[i] = rho_x * X[i - 1] + np.sqrt(max(0.0, 1.0 - rho_x ** 2)) * rng.standard_normal(d)

    V_now  = X           @ theta_true           # V*(x_t)
    V_next = np.roll(X, -1, axis=0) @ theta_true  # V*(x_{t+1}) with wrap-around

    r = V_now - gamma * V_next + sigma * rng.standard_normal(n)
    return X, r, theta_true


def generate_ar1_regression(
    n: int = 300,
    d: int = 5,
    rho: float = 0.7,
    sigma: float = 1.0,
    seed: int = 42,
):
    """
    Generate a linear regression dataset with AR(1) noise on i.i.d. features.

    Note: For i.i.d. X,  Φᵀ Φ' ≈ 0,  so LSTD(γ) ≈ OLS for any γ.
    Use `generate_mrp_data` to see LSTD's genuine advantage.

    Returns
    -------
    X, y, theta_true
    """
    rng = np.random.default_rng(seed)
    theta_true = rng.standard_normal(d)
    X = rng.standard_normal((n, d))
    eps = np.empty(n)
    eps[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        eps[i] = rho * eps[i - 1] + np.sqrt(max(0.0, 1.0 - rho ** 2)) * sigma * rng.standard_normal()
    y = X @ theta_true + eps
    return X, y, theta_true


def generate_ar1_binary(
    n: int = 300,
    d: int = 5,
    rho: float = 0.7,
    seed: int = 42,
):
    """
    Generate a binary classification dataset whose latent index has AR(1) noise
    AND AR(1) features.  Uses `generate_mrp_data` internally so that LSTD
    has a genuine structural advantage.

    Returns
    -------
    X, y (0/1), theta_true
    """
    # Use MRP-style data so correlated features make LSTD non-trivial
    X, _, theta_true = generate_mrp_data(n, d, rho_x=rho, gamma=0.0, sigma=0.0, seed=seed)
    noise, _, _ = generate_mrp_data(n, 1, rho_x=rho, gamma=0.0, sigma=0.5, seed=seed + 1)
    prob = _sigmoid(X @ theta_true + noise.ravel())
    rng  = np.random.default_rng(seed + 2)
    y    = rng.binomial(1, prob).astype(float)
    return X, y, theta_true


def compare_ols_vs_lstd(
    X_train, y_train,
    theta_true,
    gamma: float = 0.5,
    gamma_grid=None,
    cv: int = 5,
    link: str = "identity",
) -> dict:
    """
    Compare OLS (γ=0) vs LSTD(γ_CV) on coefficient recovery and prediction.

    Parameters
    ----------
    X_train, y_train : training data (typically from `generate_mrp_data`)
    theta_true       : true coefficient vector (shape (d,))
    gamma            : true discount used to generate the data (for oracle LSTD)
    gamma_grid       : grid to search over; defaults to linspace(0, 0.95, 40)
    cv               : CV folds
    link             : link function string

    Returns
    -------
    dict with keys: ols_bias, td_bias, ols_coef_err, td_coef_err,
                    best_gamma, oracle_gamma, search
    """
    if gamma_grid is None:
        gamma_grid = np.linspace(0.0, 0.95, 40)

    # ── OLS (γ=0) ────────────────────────────────────────────────────
    ols = LSTDLearner(gamma=0.0, link=link).fit(X_train, y_train)

    # ── Oracle LSTD (true γ) ─────────────────────────────────────────
    oracle = LSTDLearner(gamma=gamma, link=link).fit(X_train, y_train)

    # ── LSTD with CV-selected γ ───────────────────────────────────────
    search = GammaSearchCV(
        gamma_grid=gamma_grid,
        link=link,
        cv=cv,
        scoring="neg_mse" if link == "identity" else "neg_log_loss",
    ).fit(X_train, y_train)

    # Coefficient errors (excluding intercept if added)
    def coef_err(learner):
        t = learner.theta_
        t_coef = t[1:] if learner.add_intercept else t  # strip intercept
        return float(np.mean((t_coef - theta_true) ** 2))

    return {
        "ols_coef_err":    coef_err(ols),
        "oracle_coef_err": coef_err(oracle),
        "td_coef_err":     coef_err(search.best_estimator_),
        "best_gamma":      search.best_gamma_,
        "oracle_gamma":    gamma,
        "search":          search,
    }
