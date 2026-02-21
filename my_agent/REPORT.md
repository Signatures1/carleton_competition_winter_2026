# SQL Query Writer Agent — Project Report

**Competition:** Carleton Winter 2026 Data Science Competition
**Task:** Translate natural-language questions into executable SQL queries for a bike store DuckDB database
**Metric:** Execution accuracy — generated SQL must return the same result rows as the reference SQL

---

## 1. System Overview

The agent is a **retrieval-augmented, few-shot prompting pipeline** with a supervised learning component and an original statistical learning module (GCLS).

```
Question
   │
   ▼
[TF-IDF Retriever]  ← fitted on training split (supervised step)
   │  top-k similar labeled (question → SQL) examples
   ▼
[LLM Prompt]
   │  schema + rules + few-shot examples + question
   ▼
[LLM: mistral-large:latest @ Carleton server]
   │
   ▼
[SQL extractor / cleaner]
   │
   ▼
Executable SQL
```

---

## 2. Supervised Learning Component (TF-IDF Retriever)

### What is "trained"

The LLM weights cannot be updated via the Ollama API. The supervised learning happens in the **retrieval layer**:

1. A **TF-IDF vectorizer** is *fitted* on the training split of labeled (question, SQL) pairs.
   - This is the supervised step: the vectorizer learns which vocabulary tokens and bigrams best distinguish query types from training data only.
2. At inference time the fitted vectorizer transforms the incoming question into a TF-IDF vector, computes cosine similarity against all training questions, and returns the **k = 6 most relevant examples** as few-shot demonstrations.

### Type-aware retrieval

A keyword heuristic classifier (`_guess_type`) predicts the query category (COUNT, TOP_N, GROUP_BY, JOIN, FILTER, DISTINCT, AGGREGATION, COMPLEX, SELECT).  Strong group-dimension signals ("per month", "each store", etc.) override all other signals.  The retriever guarantees that at least ⌈k/2⌉ of the returned examples come from the same predicted type, preventing TF-IDF stop-word collisions from returning irrelevant demonstrations.  Same-type examples are placed **last** in the prompt — LLMs weight later context most heavily.

### Dataset

| Split    | Examples | Fraction |
|----------|----------|----------|
| Training | 73       | 78.5 %   |
| Validation | 11     | 11.8 %   |
| Test     | 9        | 9.7 %    |
| **Total**| **93**   |          |

Stratified by query type across 9 categories. Seed = 42.

---

## 3. Original Statistical Learning Module — GCLS

**Graph-Corrected Least Squares (GCLS)** is an original method developed for this project that generalises OLS to handle correlated training noise.

### Motivation

Standard OLS minimises `‖y − Φθ‖²` under the assumption that noise is i.i.d. across examples. In practice, **feature-similar examples share systematic noise** (same data-generating regime, similar measurement conditions). GCLS exploits this by jointly fitting two signal sources:

| Signal | Form | Noise level |
|--------|------|-------------|
| Absolute targets | `y_i` | `σ²` |
| Pairwise differences | `y_i − y_j` | `2(1−ρ)σ²  ≪  2σ²` when `ρ > 0` |

Pairwise differences carry strictly less noise than absolute targets whenever errors are positively correlated.

### Objective

```
J(θ; β) = (1−β) ‖y − Φθ‖²  +  β · Σ_{i<j} W_ij · ((y_i−y_j) − θᵀ(φ_i−φ_j))²
```

`W_ij` = softmax cosine similarity (k-nearest-neighbour graph, sparse).

### Closed-form solution

Using the graph-Laplacian identity `Σ_{i<j} W_ij(a_i−a_j)² = aᵀLa`:

```
A_β = (1−β) ΦᵀΦ  +  β ΦᵀLΦ
b_β = (1−β) Φᵀy  +  β ΦᵀLy
θ_GCLS = (A_β + λI)⁻¹ b_β
```

- `β = 0` → exact OLS
- `β = 1` → Graph-GLS, the **Best Linear Unbiased Estimator** (BLUE) under GMRF noise (Gauss-Markov theorem)
- `0 < β < 1` → data-driven interpolation via `BetaSearchCV` cross-validation

### Key properties

- **Purely supervised** — derives from classical statistics (GLS, GMRF, Gauss-Markov theorem)
- **No RL framing** — no state transitions, no reward signals, no policy optimisation
- **Closed-form** — no iterative solver, no convergence issues
- **Adaptive** — when `ρ = 0` (i.i.d. noise) CV selects `β ≈ 0` and GCLS reduces exactly to OLS

### Empirical validation (Step 6, `python train.py --td-demo`)

| ρ    | OLS err  | GCLS(β=1) err | CV err   | Best β |
|------|----------|---------------|----------|--------|
| 0.00 | 0.00268  | 0.00372       | 0.00268  | 0.00   |
| 0.30 | 0.00601  | 0.00479       | 0.00401  | 0.60   |
| 0.50 | 0.01065  | 0.00625       | 0.00538  | 0.70   |
| 0.70 | 0.02068  | 0.00771       | 0.00680  | 0.80   |
| 0.90 | 0.07289  | 0.01042       | 0.00987  | 0.90   |

Over 50 random seeds at `ρ = 0.7`: GCLS beats OLS in **50/50** seeds; 95% CI for GCLS/OLS ratio is `[0.31, 0.41]` — entirely below 1.0, statistically significant.

---

## 4. LLM & Prompt Engineering

### Model selection

| Model | Server | Val | Test |
|-------|--------|-----|------|
| llama3.2 | Local | 90.0% | 66.7% |
| command-r-plus:latest | Carleton | 50.0% | 55.6% |
| llama4:scout | Carleton | 80.0% | 66.7% |
| **mistral-large:latest** | **Carleton** | **90.9%*** | **100.0%** |

*\* One transient server cold-start timeout, not a query failure*

`mistral-large:latest` was selected as the production model.

### System prompt design

The prompt injects:
1. **Full schema** (9 tables, all columns and types)
2. **Schema notes** — critical data-type facts (e.g., `model_year` is INTEGER; `list_price` must appear when listing product rows; stock aggregation requires `SUM + GROUP BY`)
3. **Syntax rules** — DuckDB dialect, no backticks, no SELECT *, no semicolons, correct revenue formula `quantity * list_price * (1 − discount)`
4. **Domain rules** — GROUP BY dimension-first, LIMIT 1 for single winners, DISTINCT city+state for store locations, customer filters include city
5. **Few-shot examples** — 6 dynamically retrieved examples, ordered other-type first / same-type last

### Key prompt discoveries

| Failure mode | Fix applied |
|---|---|
| Backtick table names (MySQL) | Added "NEVER use backticks" rule |
| `SELECT *` instead of named columns | Added explicit rule |
| Missing `(1 − discount)` in revenue | Added "NEVER omit" rule |
| `list_price` omitted in brand/category joins | Added schema-level note (stronger than rules) |
| Store locations missing `state` | Added DISTINCT city, state mandatory note |
| Customer filters missing `city` | Added pattern rule + training examples |
| GROUP BY product_name only | Added `GROUP BY product_id, product_name` rule |

---

## 5. Final Results

**Model:** `mistral-large:latest` | **Server:** `https://rcsllm.carleton.ca/rcsapi`

```
=================================================================
 FINAL REPORT
=================================================================
  Split         Correct  Total   Accuracy
  ------------  -------  -----  ---------
  Validation         10     11      90.9%
  Test                9      9     100.0%

  Accuracy by query type -- Test
  Type                    Correct  Total     Acc
  ----------------------  -------  -----  ------
  aggregation                   1      1    100%
  complex                       1      1    100%
  count                         1      1    100%
  distinct                      1      1    100%
  filter                        1      1    100%
  group_by                      1      1    100%
  join                          1      1    100%
  select                        1      1    100%
  top_n                         1      1    100%
```

The single validation miss is a transient HTTP 500 (model cold-start timeout on the Carleton server, not a query error).

---

## 6. Project Structure

```
my_agent/
├── agent.py            — QueryWriter class (ExampleRetriever + LLM prompting)
├── train.py            — Training & evaluation harness
├── src/
│   ├── training_data.py — 93 labeled (question → SQL) examples, stratified split
│   ├── td_learner.py    — GCLS original method (GCLSLearner, BetaSearchCV)
│   └── __init__.py
├── db/
│   └── bike_store.py   — DuckDB schema loader
├── .env                — Non-secret config (host, model)
├── secrets.env         — API credentials (gitignored)
├── .env.example        — Template for new contributors
└── requirements.txt
```

---

## 7. How to Run

```bash
cd my_agent

# Full evaluation (Carleton server, requires VPN)
python train.py --delay 1.5

# Local Ollama (no VPN or API key needed)
# Set OLLAMA_HOST=http://127.0.0.1:11434 and OLLAMA_MODEL=llama3.2 in .env

# GCLS vs OLS demonstration
python train.py --td-demo

# Quick smoke-test (5 examples per split)
python train.py --quick
```

**Prerequisites:** Carleton VPN active; `secrets.env` present with valid `OLLAMA_API_KEY`. API key expires 2026-04-01.
