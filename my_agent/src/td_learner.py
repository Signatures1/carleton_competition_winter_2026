"""
Graph-Corrected Least Squares  (GCLS)
======================================

Original method — motivation
------------------------------
Standard OLS minimises  ||y - Phi theta||^2  under the assumption that noise
is i.i.d. across training examples.  In practice, examples that are *similar
in feature space* often share systematic noise — a feature-space neighbour
probably lives in the same data-generating regime, faces the same measurement
conditions, and therefore has a correlated error.

GCLS exploits this by combining two complementary signal sources:

  (1) Absolute targets     y_i          -- direct supervision, noisy
  (2) Pairwise differences y_i - y_j    -- relative supervision, LESS noisy
                                            when noise is spatially correlated

Key observation
---------------
If  Cov(eps_i, eps_j) = rho * sigma^2  for feature-similar i, j  then

    Var(eps_i - eps_j) = 2(1 - rho) sigma^2   <<   2 sigma^2   (when rho > 0)

Pairwise differences carry strictly less noise than absolute targets whenever
training examples have positively correlated errors.  GCLS harvests this
by fitting *both* the absolute targets and the pairwise differences jointly.

The GCLS objective
------------------
        J(theta; beta) =
            (1 - beta) * ||y - Phi theta||^2
          + beta       * sum_{i<j} W_ij * ((y_i-y_j) - theta^T(phi_i-phi_j))^2

where W_ij = softmax cosine similarity between phi_i and phi_j (only k-nearest
neighbours are nonzero, so the graph is sparse).

The second term is zero iff the model perfectly predicts every pairwise
difference between similar examples.  When noise is correlated within
feature-similar clusters, this term detects and corrects the bias.

Closed-form solution
---------------------
Using the graph-Laplacian identity  sum_{i<j} W_ij (a_i-a_j)^2 = a^T L a,
the objective simplifies to a weighted least-squares problem:

    J(theta; beta) = (y - Phi theta)^T Sigma_beta (y - Phi theta)

    Sigma_beta = (1 - beta) * I  +  beta * L          [precision-like matrix]
    L          = D - W                                  [graph Laplacian]
    D          = diag(row sums of W)

The unique minimiser is:

    theta_GCLS = A_beta^{-1} b_beta

    A_beta = Phi^T Sigma_beta Phi = (1-beta) Phi^T Phi  +  beta Phi^T L Phi
    b_beta = Phi^T Sigma_beta y   = (1-beta) Phi^T y    +  beta Phi^T L y

Statistical guarantee
----------------------
GCLS(beta=1) is the Best Linear Unbiased Estimator (BLUE) whenever the
noise precision matrix is proportional to L -- i.e. when the noise follows
a Gaussian Markov Random Field on the feature-similarity graph.  This
strictly dominates OLS by the Gauss-Markov theorem whenever rho > 0.

Hyperparameter selection
-------------------------
beta = 0   -> exact OLS (recovers when noise is i.i.d.)
beta = 1   -> Graph-GLS (optimal when noise ~ GMRF with precision L)
0 < beta < 1 -> interpolation selected by cross-validation (BetaSearchCV)

Original contributions
-----------------------
* Pairwise-difference objective for spatially correlated SL noise
* Feature-graph as a data-driven proxy for the noise precision matrix
* Closed-form beta-parameterised estimator unifying OLS and graph-GLS
* Purely supervised; derives from classical statistics (GLS, GMRF, Gauss-Markov)
"""

from __future__ import annotations

import warnings
import numpy as np

__all__ = [
    "build_feature_graph",
    "GCLSLearner",
    "BetaSearchCV",
    "generate_correlated_data",
    "compare_ols_vs_gcls",
]


# ============================================================================
# Section 1 -- Utilities
# ============================================================================

def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X."""
    return np.hstack([np.ones((X.shape[0], 1), dtype=float),
                      np.asarray(X, dtype=float)])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _link_fn(link: str, eta: np.ndarray):
    eta = np.asarray(eta, dtype=float)
    if link == "identity":
        return eta.copy(), np.ones_like(eta)
    if link == "logit":
        mu = _sigmoid(eta)
        mu = np.clip(mu, 1e-8, 1.0 - 1e-8)
        return mu, mu * (1.0 - mu)
    if link == "log":
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        return mu, mu
    raise ValueError(f"Unknown link '{link}'. Choose: 'identity', 'logit', 'log'.")


def _r2(y, y_hat) -> float:
    y = np.asarray(y, dtype=float)
    ss_res = np.sum((y - np.asarray(y_hat, dtype=float)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-15))


def _neg_mse(y, y_hat) -> float:
    return -float(np.mean((np.asarray(y) - np.asarray(y_hat)) ** 2))


def _neg_log_loss(y, y_hat) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(y_hat, dtype=float), 1e-10, 1.0 - 1e-10)
    return float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _score(link: str, scoring: str, y, y_hat) -> float:
    if scoring == "r2":
        return _r2(y, y_hat)
    if scoring == "neg_mse":
        return _neg_mse(y, y_hat)
    if scoring == "neg_log_loss":
        return _neg_log_loss(y, y_hat)
    raise ValueError(f"Unknown scoring '{scoring}'.")


# ============================================================================
# Section 2 -- Feature similarity graph
# ============================================================================

def build_feature_graph(
    Phi: np.ndarray,
    k: int = 5,
    temperature: float = 1.0,
) -> tuple:
    """
    Build a sparse k-nearest-neighbour feature similarity graph.

    For each training example i, the k most similar examples are found by
    cosine similarity.  Edge weights are softmax-normalised similarities,
    then symmetrised.  Returns both the adjacency matrix W and the graph
    Laplacian L = D - W.

    The Laplacian encodes the noise precision structure: if two examples
    are feature-similar (high W_ij), their noise is expected to be
    correlated, and L captures this as a Gaussian Markov Random Field
    precision matrix.

    Parameters
    ----------
    Phi         : (n, p)  feature matrix (intercept already prepended).
    k           : number of nearest neighbours per node.
    temperature : softmax temperature for edge weights.

    Returns
    -------
    W : (n, n)  symmetric adjacency matrix (row-sum = 1 before symmetrisation)
    L : (n, n)  graph Laplacian = D - W
    """
    n = Phi.shape[0]
    k = min(k, n - 1)

    # Cosine similarity matrix
    norms = np.maximum(np.linalg.norm(Phi, axis=1, keepdims=True), 1e-10)
    Phi_n = Phi / norms
    sim   = Phi_n @ Phi_n.T                          # (n, n)
    np.fill_diagonal(sim, -np.inf)                   # no self-loops

    # Top-k per row with softmax weights
    top_k  = np.argpartition(sim, -k, axis=1)[:, -k:]  # (n, k)
    rows   = np.arange(n)[:, None]
    s_top  = sim[rows, top_k] / temperature
    s_top -= s_top.max(axis=1, keepdims=True)           # numerical stability
    w      = np.exp(s_top)
    w     /= w.sum(axis=1, keepdims=True)               # (n, k), sums to 1

    # Build adjacency matrix and symmetrise
    W             = np.zeros((n, n))
    W[rows, top_k] = w
    W             = 0.5 * (W + W.T)

    # Graph Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W

    return W, L


# ============================================================================
# Section 3 -- GCLSLearner
# ============================================================================

class GCLSLearner:
    """
    Graph-Corrected Least Squares (GCLS).

    Fits a linear model by minimising a convex combination of:
      - Absolute prediction error   :  ||y - Phi theta||^2
      - Pairwise difference error   :  sum_{i<j} W_ij * ((y_i-y_j) - theta^T(phi_i-phi_j))^2

    The pairwise term is equivalent to  (y - Phi theta)^T L (y - Phi theta),
    where L is the graph Laplacian of the feature similarity graph.

    Closed-form normal equations:

        A_beta = (1-beta) Phi^T Phi  +  beta Phi^T L Phi
        b_beta = (1-beta) Phi^T y    +  beta Phi^T L y
        theta  = (A_beta + ridge)^{-1} b_beta

    When beta=0 this is exact OLS.
    When beta=1 this is graph-GLS, the BLUE under GMRF noise with precision L.

    Parameters
    ----------
    beta        : mixing parameter in [0, 1].  0 = OLS, 1 = graph-GLS.
    k           : KNN graph degree.
    temperature : softmax temperature for graph edge weights.
    link        : "identity" (regression), "logit" (classification),
                  or "log" (count / Poisson).
    add_intercept : prepend a bias column to X.
    lambda_reg  : ridge penalty for numerical stability.
    max_irls_iter, irls_tol : IRLS settings for non-identity links.
    """

    def __init__(
        self,
        beta: float = 0.0,
        k: int = 5,
        temperature: float = 1.0,
        link: str = "identity",
        add_intercept: bool = True,
        lambda_reg: float = 1e-6,
        max_irls_iter: int = 40,
        irls_tol: float = 1e-7,
    ):
        if not (0.0 <= beta <= 1.0):
            raise ValueError("beta must be in [0, 1].")
        if link not in ("identity", "logit", "log"):
            raise ValueError("link must be 'identity', 'logit', or 'log'.")

        self.beta          = beta
        self.k             = k
        self.temperature   = temperature
        self.link          = link
        self.add_intercept = add_intercept
        self.lambda_reg    = lambda_reg
        self.max_irls_iter = max_irls_iter
        self.irls_tol      = irls_tol
        self.theta_: np.ndarray | None = None
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    def _gcls_solve(
        self, Phi: np.ndarray, L: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Solve the GCLS normal equations for the identity link.

        A = (1-beta) Phi^T Phi  +  beta Phi^T L Phi
        b = (1-beta) Phi^T y    +  beta Phi^T L y
        """
        p     = Phi.shape[1]
        ridge = self.lambda_reg * np.eye(p)
        PhiTL = Phi.T @ L
        A = (1.0 - self.beta) * (Phi.T @ Phi) + self.beta * (PhiTL @ Phi) + ridge
        b = (1.0 - self.beta) * (Phi.T @ y)   + self.beta * (PhiTL @ y)
        theta, *_ = np.linalg.lstsq(A, b, rcond=None)
        return theta

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "GCLSLearner":
        """
        Fit GCLS.

        For the identity link a single linear solve is used.
        For logit / log links an IRLS loop is run where each weighted
        least-squares sub-problem uses the GCLS normal equations.

        Parameters
        ----------
        X : array-like, shape (n, d)
        y : array-like, shape (n,)
        """
        X   = np.atleast_2d(np.asarray(X, dtype=float))
        y   = np.asarray(y, dtype=float).ravel()
        Phi = _add_intercept(X) if self.add_intercept else X.copy()
        p   = Phi.shape[1]

        _, L = build_feature_graph(Phi, k=self.k, temperature=self.temperature)

        if self.link == "identity":
            self.theta_  = self._gcls_solve(Phi, L, y)
            self.n_iter_ = 1
            return self

        # ---- IRLS for non-identity links --------------------------------
        # Warm-start from pure OLS
        ridge0 = self.lambda_reg * np.eye(p)
        theta, *_ = np.linalg.lstsq(Phi.T @ Phi + ridge0, Phi.T @ y, rcond=None)
        eta = Phi @ theta

        for i in range(self.max_irls_iter):
            eta_old    = eta.copy()
            mu, W_diag = _link_fn(self.link, eta)

            # IRLS working response
            z = eta + (y - mu) / np.maximum(W_diag, 1e-10)

            # GCLS normal equations with IRLS weights
            # A = (1-beta) Phi^T diag(W) Phi  +  beta Phi^T diag(W) L Phi
            # b = (1-beta) Phi^T diag(W) z    +  beta Phi^T diag(W) L z
            WPhi  = Phi * W_diag[:, None]           # (n, p), row-scaled
            WLPhi = (W_diag[:, None] * (L @ Phi))   # (n, p)
            ridge_irls = max(self.lambda_reg, 1e-4) * np.eye(p)
            A = ((1.0 - self.beta) * (WPhi.T @ Phi)
                 + self.beta       * (WLPhi.T @ Phi)
                 + ridge_irls)
            b = ((1.0 - self.beta) * (WPhi.T @ z)
                 + self.beta       * (WLPhi.T @ z))

            theta_new, *_ = np.linalg.lstsq(A, b, rcond=None)

            # Backtracking: reject step if eta norm explodes
            eta_prop = Phi @ theta_new
            alpha = 1.0
            prev_norm = np.linalg.norm(eta_old)
            for _ in range(8):
                eta_cand = (1.0 - alpha) * eta_old + alpha * eta_prop
                if np.linalg.norm(eta_cand) <= 4.0 * max(prev_norm, 1.0):
                    break
                alpha *= 0.5
            else:
                eta_cand = eta_old
            theta = (1.0 - alpha) * theta + alpha * theta_new
            eta   = eta_cand

            delta = np.max(np.abs(eta - eta_old))
            if delta < self.irls_tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_irls_iter
            if delta > 1.0:
                warnings.warn(
                    f"GCLS-IRLS did not converge after {self.max_irls_iter} "
                    f"iterations (delta_eta = {delta:.3e}).",
                    RuntimeWarning,
                )

        self.theta_ = theta
        return self

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """Return predictions in response space."""
        if self.theta_ is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.add_intercept:
            X = _add_intercept(X)
        mu, _ = _link_fn(self.link, X @ self.theta_)
        return mu

    # ------------------------------------------------------------------
    def score(self, X, y) -> float:
        """R2 (identity), neg-log-loss (logit), neg-MSE (log)."""
        y_hat = self.predict(X)
        if self.link == "identity":
            return _r2(y, y_hat)
        if self.link == "logit":
            return _neg_log_loss(y, y_hat)
        return _neg_mse(y, y_hat)


# ============================================================================
# Section 4 -- BetaSearchCV
# ============================================================================

class BetaSearchCV:
    """
    Cross-validated grid search over the GCLS mixing parameter beta.

    Searches beta in [0, 1] and selects the value that maximises a
    held-out score metric on k folds, then re-fits on all data.

    beta = 0   -> pure OLS
    beta = 1   -> graph-GLS (optimal under GMRF noise)
    CV-selected beta provides a fully data-driven estimator.

    Parameters
    ----------
    beta_grid   : 1-D array of beta candidates.  Default: 11 values in [0, 1].
    k           : KNN graph degree.
    temperature : edge-weight softmax temperature.
    link        : link function.
    cv          : number of cross-validation folds.
    scoring     : "neg_mse", "r2", or "neg_log_loss".
    add_intercept, lambda_reg : passed to GCLSLearner.
    verbose     : print per-beta CV scores.
    """

    def __init__(
        self,
        beta_grid=None,
        k: int = 5,
        temperature: float = 1.0,
        link: str = "identity",
        cv: int = 5,
        scoring: str = "neg_mse",
        add_intercept: bool = True,
        lambda_reg: float = 1e-6,
        verbose: bool = False,
    ):
        self.beta_grid = (
            np.linspace(0.0, 1.0, 11)
            if beta_grid is None
            else np.asarray(beta_grid, dtype=float)
        )
        self.k             = k
        self.temperature   = temperature
        self.link          = link
        self.cv            = cv
        self.scoring       = scoring
        self.add_intercept = add_intercept
        self.lambda_reg    = lambda_reg
        self.verbose       = verbose

        self.best_beta_:      float | None        = None
        self.best_score_:     float | None        = None
        self.cv_results_:     dict  | None        = None
        self.best_estimator_: GCLSLearner | None  = None

    def fit(self, X, y) -> "BetaSearchCV":
        """Run CV grid search and refit best model."""
        from sklearn.model_selection import KFold

        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        kf = KFold(n_splits=min(self.cv, len(y)), shuffle=False)

        mean_scores = np.empty(len(self.beta_grid))
        std_scores  = np.empty(len(self.beta_grid))

        for b_idx, beta in enumerate(self.beta_grid):
            fold_scores = []
            for tr, va in kf.split(X):
                m = GCLSLearner(
                    beta=float(beta), k=self.k, temperature=self.temperature,
                    link=self.link, add_intercept=self.add_intercept,
                    lambda_reg=self.lambda_reg,
                ).fit(X[tr], y[tr])
                fold_scores.append(
                    _score(self.link, self.scoring, y[va], m.predict(X[va]))
                )
            mean_scores[b_idx] = np.mean(fold_scores)
            std_scores[b_idx]  = np.std(fold_scores)
            if self.verbose:
                print(f"  beta={beta:.2f}  mean_{self.scoring}="
                      f"{mean_scores[b_idx]:+.5f} +/- {std_scores[b_idx]:.5f}")

        best_idx         = int(np.argmax(mean_scores))
        self.best_beta_  = float(self.beta_grid[best_idx])
        self.best_score_ = float(mean_scores[best_idx])
        self.cv_results_ = {
            "beta":       self.beta_grid,
            "mean_score": mean_scores,
            "std_score":  std_scores,
        }
        self.best_estimator_ = GCLSLearner(
            beta=self.best_beta_, k=self.k, temperature=self.temperature,
            link=self.link, add_intercept=self.add_intercept,
            lambda_reg=self.lambda_reg,
        ).fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.best_estimator_.predict(X)

    def plot_beta_curve(self, ax=None):
        """Plot mean +/- std CV score vs beta.  Requires matplotlib."""
        import matplotlib.pyplot as plt
        if self.cv_results_ is None:
            raise RuntimeError("Call fit() before plot_beta_curve().")
        r = self.cv_results_
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r["beta"], r["mean_score"], color="steelblue", linewidth=2,
                label=f"CV {self.scoring}")
        ax.fill_between(r["beta"],
                        r["mean_score"] - r["std_score"],
                        r["mean_score"] + r["std_score"],
                        alpha=0.2, color="steelblue", label="+/- 1 SD")
        ax.axvline(self.best_beta_, color="crimson", linestyle="--",
                   label=f"best beta = {self.best_beta_:.2f}")
        ax.set_xlabel("beta  (graph-correction strength)")
        ax.set_ylabel(self.scoring)
        ax.set_title("GCLS -- cross-validated score vs. beta")
        ax.legend()
        return ax


# ============================================================================
# Section 5 -- Synthetic data generators
# ============================================================================

def generate_correlated_data(
    n: int = 300,
    d: int = 5,
    rho: float = 0.7,
    sigma: float = 0.5,
    k_graph: int = 10,
    seed: int = 42,
):
    """
    Generate regression data where noise is spatially correlated with
    the feature similarity structure.

    Model
    -----
        x_{t+1} = rho * x_t + sqrt(1 - rho^2) * eta_t   [AR(1) features]
        y_i     = theta*^T x_i + eps_i                    [linear signal]
        eps     ~ N(0, sigma^2 * (L_data + ridge)^{-1})   [GMRF noise]

    Because the feature graph L_data is built from the AR(1) features,
    noise is positively correlated between temporally close (similar) examples.
    This is exactly the structure GCLS is designed to exploit.

    OLS ignores this covariance and produces a biased estimator.
    GCLS(beta=1) uses L as the precision matrix and recovers theta* as the BLUE.

    Returns
    -------
    X          : (n, d)
    y          : (n,)
    theta_true : (d,)
    """
    rng = np.random.default_rng(seed)
    theta_true = rng.standard_normal(d)

    # AR(1) feature matrix
    X = np.zeros((n, d))
    X[0] = rng.standard_normal(d)
    for i in range(1, n):
        X[i] = (rho * X[i - 1]
                + np.sqrt(max(0.0, 1.0 - rho ** 2)) * rng.standard_normal(d))

    # Build graph Laplacian from features, then invert to get noise covariance
    Phi = _add_intercept(X)
    _, L = build_feature_graph(Phi, k=k_graph)

    # Regularise L before inverting (L has a zero eigenvalue)
    Sigma = np.linalg.inv(L + 1e-3 * np.eye(n)) * sigma ** 2
    Sigma = 0.5 * (Sigma + Sigma.T)   # enforce symmetry numerically

    eps = rng.multivariate_normal(np.zeros(n), Sigma)
    y   = X @ theta_true + eps

    return X, y, theta_true


# ============================================================================
# Section 6 -- Comparison utility
# ============================================================================

def compare_ols_vs_gcls(
    X_train,
    y_train,
    theta_true,
    beta_grid=None,
    cv: int = 5,
    k: int = 10,
) -> dict:
    """
    Compare OLS (beta=0) vs GCLS(beta=1) vs GCLS(beta=CV) on coefficient recovery.

    Parameters
    ----------
    X_train, y_train : training data from `generate_correlated_data`
    theta_true       : true coefficient vector shape (d,)
    beta_grid        : grid to search over (default: 11 values in [0, 1])
    cv               : CV folds
    k                : KNN graph degree

    Returns
    -------
    dict with keys: ols_coef_err, graph_gls_coef_err, cv_coef_err,
                    best_beta, search
    """
    if beta_grid is None:
        beta_grid = np.linspace(0.0, 1.0, 11)

    def _ce(theta_hat):
        return float(np.mean((theta_hat - theta_true) ** 2))

    ols      = GCLSLearner(beta=0.0, k=k).fit(X_train, y_train)
    graph_gl = GCLSLearner(beta=1.0, k=k).fit(X_train, y_train)
    search   = BetaSearchCV(beta_grid=beta_grid, k=k, cv=cv).fit(X_train, y_train)

    def _strip(m):
        t = m.theta_
        return t[1:] if m.add_intercept else t

    return {
        "ols_coef_err":       _ce(_strip(ols)),
        "graph_gls_coef_err": _ce(_strip(graph_gl)),
        "cv_coef_err":        _ce(_strip(search.best_estimator_)),
        "best_beta":          search.best_beta_,
        "search":             search,
    }
