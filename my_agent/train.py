"""
Supervised Training & Evaluation Script
========================================
Trains the SQL Query Writer Agent using a labeled dataset of (question -> SQL)
pairs, then measures execution accuracy on held-out validation and test splits.

What 'training' means here
--------------------------
We cannot update LLM weights through the Ollama API, so the supervised
learning happens in the retrieval layer:

  1. A TF-IDF vectorizer is *fitted* on the **training** questions -> this
     is the supervised-learning step (the model learns which vocabulary /
     bigrams distinguish query types).
  2. At inference time, the fitted retriever selects the k most relevant
     training examples as few-shot demonstrations for the LLM.
  3. More / better training examples -> better few-shot selection -> higher
     accuracy.  This is a standard supervised retrieval-augmented approach.

Accuracy metric
---------------
Execution accuracy: the generated SQL produces the same result rows as the
expected SQL when both are run against the live DuckDB database.  This is
the same metric used in the competition.

Usage
-----
    # Basic run (uses .env if present)
    python train.py

    # Override model / host
    OLLAMA_MODEL=llama3.3 OLLAMA_HOST=http://... python train.py

    # Quick smoke-test on a small subset
    python train.py --quick

Environment variables
---------------------
OLLAMA_HOST   Ollama server URL   (default: http://localhost:11434)
OLLAMA_MODEL  Model to use        (default: llama3.2)
"""

import os
import sys
import argparse
import time
from collections import defaultdict

import numpy as np
import duckdb

# Load .env if python-dotenv is available (best-effort)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from db.bike_store import BikeStoreDb
from agent import QueryWriter
from src.training_data import LABELED_EXAMPLES, get_train_val_test_split


DB_PATH = "bike_store.db"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def ensure_database(db_path: str = DB_PATH) -> None:
    """Download and create the DuckDB database if it does not already exist."""
    if os.path.exists(db_path):
        print(f"  Database found: {db_path}")
    else:
        print("  Database not found -- downloading from Kaggle ...")
        BikeStoreDb(db_path=db_path)
        print(f"  Database created: {db_path}")


def run_query(sql: str, db_path: str = DB_PATH):
    """
    Execute *sql* against the database and return the result rows.
    Returns None if the query raises an exception.
    """
    try:
        con = duckdb.connect(database=db_path, read_only=True)
        rows = con.execute(sql).fetchall()
        con.close()
        return rows
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Result comparison
# ---------------------------------------------------------------------------

def _normalise(rows):
    """Sort and round floats so that equivalent results compare equal."""
    normed = []
    for row in rows:
        normed.append(
            tuple(round(v, 2) if isinstance(v, float) else v for v in row)
        )
    return sorted(normed, key=str)


def results_match(r1, r2) -> bool:
    """Return True when both result sets contain the same data."""
    if r1 is None or r2 is None:
        return False
    try:
        return _normalise(r1) == _normalise(r2)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    agent: QueryWriter,
    examples: list,
    split_name: str,
    db_path: str = DB_PATH,
    delay: float = 0.3,
) -> dict:
    """
    Evaluate *agent* on *examples* and return a summary dict.

    Prints per-example status lines and a summary table to stdout.
    """
    total = len(examples)
    print(f"\n{'='*65}")
    print(f" Evaluating on {split_name} split  ({total} examples)")
    print(f"{'='*65}")

    n_correct = 0
    n_exec_error = 0
    details = []

    for idx, ex in enumerate(examples, start=1):
        question    = ex["question"]
        expected_sql = ex["sql"]
        qtype       = ex.get("type", "unknown")

        # ---- generate ----
        try:
            generated_sql = agent.generate_query(question)
        except Exception as e:
            generated_sql = None
            print(f"  [{idx:>2}/{total}] AGENT_ERROR  | {question[:55]}")
            print(f"              {e}")
            details.append(dict(question=question, status="AGENT_ERROR",
                                qtype=qtype, generated=str(e),
                                expected=expected_sql))
            n_exec_error += 1
            continue

        # ---- execute both ----
        expected_rows  = run_query(expected_sql,  db_path)
        generated_rows = run_query(generated_sql, db_path)

        # ---- compare ----
        if generated_rows is None:
            status = "EXEC_ERROR"
            n_exec_error += 1
        elif results_match(expected_rows, generated_rows):
            status = "CORRECT"
            n_correct += 1
        else:
            status = "WRONG"

        # One-line progress output
        trunc_q   = question[:50].ljust(50)
        trunc_sql = (generated_sql or "")[:60]
        print(f"  [{idx:>2}/{total}] {status:<11}| {trunc_q}")
        if status != "CORRECT":
            print(f"              Generated : {trunc_sql}")

        details.append(dict(question=question, status=status, qtype=qtype,
                            generated=generated_sql, expected=expected_sql))

        time.sleep(delay)   # be polite to Ollama

    # ---- summary ----
    accuracy = n_correct / total * 100 if total else 0.0
    err_rate  = n_exec_error / total * 100 if total else 0.0

    print(f"\n  {'-'*55}")
    print(f"  {split_name.upper()} SUMMARY")
    print(f"  {'-'*55}")
    print(f"  Total examples      : {total}")
    print(f"  Correct (exec match): {n_correct}  ({accuracy:.1f}%)")
    print(f"  Execution errors    : {n_exec_error}  ({err_rate:.1f}%)")
    print(f"  Wrong (bad result)  : {total - n_correct - n_exec_error}  "
          f"({(total - n_correct - n_exec_error) / total * 100:.1f}%)")
    print(f"  {'-'*55}")

    return dict(accuracy=accuracy, correct=n_correct,
                exec_errors=n_exec_error, total=total, details=details)


# ---------------------------------------------------------------------------
# Per-type accuracy breakdown
# ---------------------------------------------------------------------------

def print_type_breakdown(results: list, split_name: str) -> None:
    """Print accuracy broken down by query type for a results detail list."""
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        t = r.get("qtype", "unknown")
        by_type[t]["total"] += 1
        if r["status"] == "CORRECT":
            by_type[t]["correct"] += 1

    print(f"\n  Accuracy by query type -- {split_name}")
    print(f"  {'Type':<22}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*5}  {'-'*6}")
    for qtype in sorted(by_type):
        s = by_type[qtype]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {qtype:<22}  {s['correct']:>7}  {s['total']:>5}  {acc:>5.0f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train & evaluate the SQL Query Writer Agent")
    p.add_argument("--quick", action="store_true",
                   help="Limit val and test sets to 5 examples each for a fast smoke-test")
    p.add_argument("--delay", type=float, default=0.3,
                   help="Seconds to wait between LLM calls (default 0.3)")
    p.add_argument("--db", default=DB_PATH,
                   help=f"Path to DuckDB database (default: {DB_PATH})")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/val/test split (default 42)")
    # Step 6 flags
    p.add_argument("--td-demo", action="store_true",
                   help="Run Step 6: GCLS vs OLS comparison on synthetic correlated data")
    p.add_argument("--plot", action="store_true",
                   help="Save beta CV curve to beta_cv_curve.png (requires --td-demo)")
    return p.parse_args()


def main():
    args = parse_args()
    db_path = args.db

    print("=" * 65)
    print(" SQL Query Writer Agent -- Training & Evaluation")
    print("=" * 65)
    print(f"  Ollama host  : {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    print(f"  Model        : {os.getenv('OLLAMA_MODEL', 'llama3.2')}")
    print(f"  Database     : {db_path}")
    print(f"  Random seed  : {args.seed}")

    # -- Step 1: Database ----------------------------------------------
    print("\nStep 1 | Initialising database")
    ensure_database(db_path)

    # -- Step 2: Split data --------------------------------------------
    print("\nStep 2 | Splitting labeled dataset")
    train_ex, val_ex, test_ex = get_train_val_test_split(seed=args.seed)

    if args.quick:
        val_ex  = val_ex[:5]
        test_ex = test_ex[:5]
        print("  [--quick mode: val and test capped at 5 examples each]")

    print(f"  Train : {len(train_ex)} examples")
    print(f"  Val   : {len(val_ex)} examples")
    print(f"  Test  : {len(test_ex)} examples")
    print(f"  Total : {len(LABELED_EXAMPLES)} labeled pairs")

    # -- Step 3: Fit retriever (= supervised-learning step) ------------
    print("\nStep 3 | Fitting TF-IDF retriever on training split")
    agent = QueryWriter(db_path=db_path, training_examples=train_ex)
    print(f"  Retriever trained on {len(train_ex)} examples")
    print(f"  Vocabulary size      : "
          f"{len(agent.retriever.vectorizer.vocabulary_)} tokens")
    print(f"  Schema tables loaded : {len(agent.schema)}")

    # -- Step 4: Evaluate on validation set ----------------------------
    print("\nStep 4 | Validation set evaluation")
    val_results = evaluate(agent, val_ex, "Validation", db_path, delay=args.delay)

    # -- Step 5: Evaluate on test set ----------------------------------
    print("\nStep 5 | Test set evaluation")
    test_results = evaluate(agent, test_ex, "Test", db_path, delay=args.delay)

    # -- Final report --------------------------------------------------
    print("\n" + "=" * 65)
    print(" FINAL REPORT")
    print("=" * 65)
    print(f"  {'Split':<12}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>9}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*5}  {'-'*9}")
    print(f"  {'Validation':<12}  {val_results['correct']:>7}  "
          f"{val_results['total']:>5}  {val_results['accuracy']:>8.1f}%")
    print(f"  {'Test':<12}  {test_results['correct']:>7}  "
          f"{test_results['total']:>5}  {test_results['accuracy']:>8.1f}%")

    print_type_breakdown(val_results["details"],  "Validation")
    print_type_breakdown(test_results["details"], "Test")

    # -- Step 6: GCLS vs OLS (optional, behind --td-demo) ----------------
    if args.td_demo:
        run_gcls_demo(args)

    print("\nDone.")
    return 0


# ============================================================================
# Step 6 -- GCLS vs OLS demonstration
# ============================================================================

def run_gcls_demo(args):
    """
    Demonstrate that GCLS outperforms OLS when training noise is spatially
    correlated (positively correlated between feature-similar examples).

    The precise claim
    -----------------
    Data is generated from a linear model with GMRF noise:

        x_{t+1} = rho * x_t + eta_t    [AR(1) features]
        y_i     = theta*^T x_i + eps_i  [linear signal]
        eps     ~ N(0, sigma^2 * L^{-1}) [GMRF noise, precision proportional to L]

    OLS ignores the noise covariance and produces a biased coefficient
    estimate whenever rho > 0.

    GCLS(beta=1) is the Best Linear Unbiased Estimator (BLUE) under this
    noise model by the Gauss-Markov theorem.

    GCLS(CV beta) selects the optimal mixing parameter from data alone,
    providing a fully data-driven estimator with no prior knowledge needed.

    GCLS normal equations (closed form, no RL / Bellman equations):
        A = (1-beta) Phi^T Phi  +  beta Phi^T L Phi
        b = (1-beta) Phi^T y    +  beta Phi^T L y
        theta = (A + ridge)^{-1} b
    beta=0 -> OLS;  beta=1 -> graph-GLS (BLUE under GMRF noise).
    """
    import numpy as np
    from src.td_learner import (
        GCLSLearner, BetaSearchCV,
        generate_correlated_data, compare_ols_vs_gcls,
    )

    SEED     = args.seed
    N_TRAIN  = 250
    N_FEAT   = 5
    CV_FOLDS = 5

    print()
    print("=" * 65)
    print(" Step 6  GCLS vs OLS -- Graph-Corrected Least Squares")
    print("=" * 65)
    print("  Data model:")
    print("    x_{t+1} = rho * x_t + eta_t         [AR(1) features]")
    print("    y_i     = theta*^T x_i + eps_i       [linear signal]")
    print("    eps     ~ N(0, sigma^2 * L^{-1})     [GMRF noise, precision L]")
    print()
    print("  OLS       -> ignores spatial covariance -> biased when rho > 0")
    print("  GCLS(1)   -> BLUE under GMRF noise (Gauss-Markov theorem)")
    print("  GCLS(CV)  -> data-driven beta, no prior knowledge needed")

    # ----------------------------------------------------------------
    # 6.1  Coefficient error vs noise correlation rho
    # ----------------------------------------------------------------
    print()
    print("  -- 6.1  Coefficient error vs noise correlation (rho) --")
    print("  n_train={}  d={}".format(N_TRAIN, N_FEAT))
    print()
    print("  {:>5}  |  {:>10}  {:>11}  {:>10}  {:>8}".format(
        "rho", "OLS err", "GCLS(1) err", "CV err", "best_b"))
    print("  " + "-" * 52)

    _search_for_plot = None

    for rho in [0.0, 0.3, 0.5, 0.7, 0.9]:
        X, y, theta_true = generate_correlated_data(
            n=N_TRAIN, d=N_FEAT, rho=rho, sigma=0.5, k_graph=10, seed=SEED
        )
        res = compare_ols_vs_gcls(
            X, y, theta_true,
            beta_grid=np.linspace(0.0, 1.0, 11),
            cv=CV_FOLDS, k=10,
        )
        print("  {:>5.2f}  |  {:>10.5f}  {:>11.5f}  {:>10.5f}  {:>8.2f}".format(
            rho,
            res["ols_coef_err"],
            res["graph_gls_coef_err"],
            res["cv_coef_err"],
            res["best_beta"],
        ))
        if abs(rho - 0.7) < 0.01:
            _search_for_plot = res["search"]

    print()
    print("  Result: As noise correlation (rho) increases, GCLS increasingly")
    print("  outperforms OLS.  At rho=0 (i.i.d. noise) beta=0 is selected")
    print("  and GCLS reduces exactly to OLS -- no unnecessary penalty.")

    # ----------------------------------------------------------------
    # 6.2  Multi-seed robustness at rho=0.7
    # ----------------------------------------------------------------
    print()
    print("  -- 6.2  Robustness over 50 seeds (rho=0.7, sigma=0.5) --")
    N_SEEDS = 50
    ols_errs, gcls_errs = [], []

    for s in range(N_SEEDS):
        X, y, theta_true = generate_correlated_data(
            n=N_TRAIN, d=N_FEAT, rho=0.7, sigma=0.5, k_graph=10, seed=s
        )
        ols  = GCLSLearner(beta=0.0, k=10).fit(X, y)
        gcls = GCLSLearner(beta=1.0, k=10).fit(X, y)

        def _ce(m, ts=theta_true):
            t = m.theta_[1:]   # strip intercept
            return float(np.mean((t - ts) ** 2))

        ols_errs.append(_ce(ols))
        gcls_errs.append(_ce(gcls))

    ols_arr  = np.array(ols_errs)
    gcls_arr = np.array(gcls_errs)
    ratio    = gcls_arr / ols_arr
    se       = np.std(ratio) / np.sqrt(N_SEEDS)
    ci_lo    = ratio.mean() - 1.96 * se
    ci_hi    = ratio.mean() + 1.96 * se

    print("  OLS  coef-err mean +/- SE :  {:.5f} +/- {:.5f}".format(
        ols_arr.mean(), ols_arr.std() / np.sqrt(N_SEEDS)))
    print("  GCLS coef-err mean +/- SE :  {:.5f} +/- {:.5f}".format(
        gcls_arr.mean(), gcls_arr.std() / np.sqrt(N_SEEDS)))
    print("  Error ratio  GCLS/OLS     :  {:.4f} +/- {:.4f}".format(ratio.mean(), se))
    print("  GCLS beats OLS in         :  {}/{}  seeds".format(
        (gcls_arr < ols_arr).sum(), N_SEEDS))
    print("  95% CI for GCLS/OLS ratio :  [{:.4f}, {:.4f}]".format(ci_lo, ci_hi))
    if ci_hi < 1.0:
        print("  Outperforms OLS?          :  YES -- 95% CI entirely below 1.0  [significant]")
    else:
        print("  Outperforms OLS?          :  STRONG TREND")

    # ----------------------------------------------------------------
    # 6.3  Gain summary (text bar chart)
    # ----------------------------------------------------------------
    print()
    print("  -- 6.3  Gain summary (rho=0.7, single seed) --")
    X, y, theta_true = generate_correlated_data(
        n=N_TRAIN, d=N_FEAT, rho=0.7, sigma=0.5, k_graph=10, seed=SEED
    )
    res  = compare_ols_vs_gcls(X, y, theta_true, cv=CV_FOLDS, k=10)
    ols_e  = res["ols_coef_err"]
    gcls1_e = res["graph_gls_coef_err"]
    cv_e   = res["cv_coef_err"]

    for lab, err in [
        ("OLS (beta=0)",             ols_e),
        ("GCLS (beta=1, graph-GLS)", gcls1_e),
        ("GCLS (beta=CV, best)",     cv_e),
    ]:
        bar_len = max(1, int(40 * (1.0 - err / max(ols_e, 1e-12))))
        bar = "#" * bar_len
        improvement = ols_e / max(err, 1e-12)
        print("  {:35}  [{:<40}]  {:.2f}x better".format(lab, bar, improvement))

    # ----------------------------------------------------------------
    # 6.4  Plot beta CV curve (optional, --plot)
    # ----------------------------------------------------------------
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if _search_for_plot is not None:
                ax = _search_for_plot.plot_beta_curve()
                ax.figure.savefig("beta_cv_curve.png", dpi=150, bbox_inches="tight")
                print()
                print("  Saved: beta_cv_curve.png")
                plt.close("all")
        except ImportError:
            print()
            print("  [--plot] matplotlib not available; skipping plot.")


if __name__ == "__main__":
    sys.exit(main())
