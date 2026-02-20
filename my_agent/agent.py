"""
SQL Query Writer Agent

Implements the QueryWriter class that translates natural language questions
into executable SQL queries for the bike store DuckDB database.

Strategy
--------
1. Build a detailed system prompt that includes the full database schema.
2. Retrieve the k most semantically similar labeled examples from the training
   set (TF-IDF cosine similarity) and inject them as few-shot demonstrations.
3. Ask the LLM to generate a SQL query, then post-process / extract it cleanly.

Environment variables
---------------------
OLLAMA_HOST   - Ollama server URL   (default: http://localhost:11434)
OLLAMA_MODEL  - Model to use        (default: llama3.2)
"""

import os
import re

from db.bike_store import get_schema_info

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def get_ollama_client():
    """
    Return an Ollama client pointed at either the Carleton server or localhost.

    Set OLLAMA_HOST to override the default local instance.
    """
    import ollama
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return ollama.Client(host=host)


def get_model_name() -> str:
    """Return the model name from env, defaulting to llama3.2."""
    return os.getenv("OLLAMA_MODEL", "llama3.2")


# ---------------------------------------------------------------------------
# Few-shot example retriever
# ---------------------------------------------------------------------------

# Keyword signals for each query type.  Multiple matches → higher confidence.
_TYPE_SIGNALS: dict = {
    "count":       ["how many", "total number of", "count the number"],
    "distinct":    ["what states", "what cities", "what model year", "what years",
                    "distinct", "unique values"],
    "top_n":       ["top 5", "top 10", "top 3", "top 1", "most expensive",
                    "cheapest", "most recent", "least expensive", "best ",
                    "lowest priced", "highest priced", "3 most", "5 most",
                    "10 most", "10 least"],
    "aggregation": ["average", "total revenue", "maximum", "minimum",
                    "most inventory", "highest stock", "what is the max",
                    "what is the min", "what is the avg", "what is the total"],
    "group_by":    ["per store", "per brand", "per category", "per year",
                    "per month", "each store", "each brand", "each category",
                    "each year", "each month", "by brand", "by category",
                    "by store", "by year", "by month", "how many orders per",
                    "how many products per"],
    "filter":      ["under $", "over $", "less than $", "more than $",
                    "from 2018", "from 2019", "in 2018", "in 2019", "active",
                    "completed", "new york", "model year 20", "trek",
                    "priced between", "price range"],
    "complex":     ["never been ordered", "out of stock", "most money",
                    "best-selling", "has handled", "has placed", "has spent",
                    "more than 2 orders", "not been", "have never"],
    "join":        ["with their brand", "with their category", "with their store",
                    "with order dates", "with customer names", "and their store",
                    "and brand", "and category", "with product names"],
    "select":      ["list all", "show all", "list the", "show the"],
}


class ExampleRetriever:
    """
    Given a pool of labeled (question, sql) examples, retrieves the k most
    semantically relevant examples for a new query using a hybrid of
    TF-IDF cosine similarity and query-type-aware selection.

    This is the 'supervised learning' component: the TF-IDF vectorizer is
    *fitted* on the training split, and all downstream retrievals are made
    relative to that learned vocabulary.

    Type-aware selection guarantees that at least half of the returned
    examples come from the predicted query type, preventing TF-IDF
    stop-word collisions from returning completely irrelevant examples.
    """

    def __init__(self, examples: list):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        self._cosine_similarity = cosine_similarity
        self._np = np
        self.examples = examples

        questions = [ex["question"] for ex in examples]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=None,        # keep all tokens: "how many" is diagnostic
            sublinear_tf=True,
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(questions)

    # ------------------------------------------------------------------
    def _guess_type(self, query: str) -> str | None:
        """
        Keyword-heuristic classifier returning the most likely query type.
        Returns None when no signal fires (falls back to pure TF-IDF).
        """
        q = query.lower()
        # Strong override: "each/per <time|dimension>" always means GROUP BY,
        # even when "how many" is also present.
        _group_overrides = (
            "each month", "each year", "each store", "each brand",
            "each category", "per month", "per year", "per store",
            "per brand", "per category",
        )
        if any(sig in q for sig in _group_overrides):
            return "group_by"
        scores = {t: sum(1 for sig in sigs if sig in q)
                  for t, sigs in _TYPE_SIGNALS.items()}
        best, best_score = max(scores.items(), key=lambda kv: kv[1])
        return best if best_score > 0 else None

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 6) -> list:
        """
        Return the k most relevant examples for *query*.

        Strategy
        --------
        1. TF-IDF cosine similarity ranks all training examples.
        2. Keyword heuristics guess the query type.
        3. At least ceil(k/2) slots are filled from same-type examples
           (ranked by TF-IDF); the remaining slots come from the global
           top-similarity examples regardless of type.
        4. Final list is re-sorted by TF-IDF score for coherent ordering.
        """
        query_vec = self.vectorizer.transform([query])
        sims = self._cosine_similarity(query_vec, self.tfidf_matrix)[0]
        all_ranked = self._np.argsort(sims)[::-1]

        guessed = self._guess_type(query)
        if guessed:
            n_type  = max(2, (k + 1) // 2)
            same    = [i for i in all_ranked
                       if self.examples[i].get("type") == guessed]
            other   = [i for i in all_ranked
                       if self.examples[i].get("type") != guessed]
            n_type  = min(n_type, len(same))
            n_other = k - n_type
            # Present other-type examples first, same-type examples last.
            # LLMs weight later (closer-to-question) examples most heavily,
            # so same-type examples at the end have maximum influence.
            same_top  = sorted(same[:n_type],  key=lambda i: sims[i], reverse=True)
            other_top = sorted(other[:n_other], key=lambda i: sims[i], reverse=True)
            picked = other_top + same_top
        else:
            picked = list(all_ranked[:k])

        return [self.examples[i] for i in picked[:k]]


# ---------------------------------------------------------------------------
# QueryWriter
# ---------------------------------------------------------------------------

_SCHEMA_NOTES = """
Key relationships:
  products.brand_id    → brands.brand_id
  products.category_id → categories.category_id
  orders.customer_id   → customers.customer_id
  orders.store_id      → stores.store_id
  orders.staff_id      → staffs.staff_id
  order_items.order_id → orders.order_id
  order_items.product_id → products.product_id
  stocks.store_id      → stores.store_id
  stocks.product_id    → products.product_id
  staffs.store_id      → stores.store_id

orders.order_status values: 1=Pending, 2=Processing, 3=Rejected, 4=Completed
staffs.active values:       1=Active, 0=Inactive

IMPORTANT column notes:
  products.model_year is an INTEGER (e.g. 2016, 2017, 2018, 2019).
    Use WHERE model_year = 2018 — do NOT use EXTRACT() on this column.
  order_date, required_date, shipped_date are DATE columns.
    Use EXTRACT(YEAR FROM order_date) = 2019 to filter by year.
""".strip()


class QueryWriter:
    """
    SQL Query Writer Agent — competition entry point.

    The `generate_query` method is called by the evaluation harness with a
    natural-language question and must return a clean, executable SQL string.
    """

    def __init__(self, db_path: str = "bike_store.db", training_examples: list = None):
        """
        Args:
            db_path:           Path to the DuckDB database file.
            training_examples: Optional list of labeled examples used to prime
                               the few-shot retriever.  When None the module
                               attempts to load them from src.training_data.
        """
        self.db_path = db_path
        self.schema = get_schema_info(db_path=db_path)
        self.client = get_ollama_client()
        self.model = get_model_name()
        self._schema_text = self._format_schema()

        # Build the few-shot retriever from training examples
        if training_examples is not None:
            self.retriever = ExampleRetriever(training_examples)
        else:
            try:
                from src.training_data import get_train_val_test_split
                train_examples, _, _ = get_train_val_test_split()
                self.retriever = ExampleRetriever(train_examples)
            except Exception:
                self.retriever = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_query(self, prompt: str) -> str:
        """
        Translate a natural-language question into a SQL query.

        Args:
            prompt: Natural language question, e.g.
                    "What are the top 5 most expensive products?"

        Returns:
            A valid SQL string ready to execute against the bike store DB.
        """
        few_shot_block = self._build_few_shot_block(prompt)
        system_prompt = self._build_system_prompt(few_shot_block)

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Question: {prompt}"},
            ],
            options={"temperature": 0.0},   # deterministic output
        )

        raw = response["message"]["content"].strip()
        return self._extract_sql(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, few_shot_block: str) -> str:
        parts = [
            "You are an expert SQL query writer for a bike store relational database.",
            "",
            "=== DATABASE SCHEMA ===",
            self._schema_text,
            "",
            _SCHEMA_NOTES,
            "",
            "=== RULES ===",
            "- Return ONLY the raw SQL query — no explanations, no markdown, no code fences.",
            "- Do NOT wrap the query in ```sql ... ``` or any other delimiters.",
            "- Use the exact column and table names from the schema above.",
            "- Use proper table aliases in JOIN queries.",
            "- Use DuckDB-compatible syntax.",
            "- For year/month extraction use EXTRACT(YEAR FROM col) / EXTRACT(MONTH FROM col).",
            "- For revenue calculations use: quantity * list_price * (1 - discount).",
            "- For COUNT queries ('how many'), always use COUNT(*) with an alias.",
            "- For DISTINCT value queries ('what X exist / are available'), use SELECT DISTINCT.",
            "- Do NOT add DISTINCT unless the question explicitly asks for unique or distinct values.",
            "- Always add ORDER BY when the question implies ranking or a list.",
            "- For GROUP BY queries, include all non-aggregated SELECT columns in GROUP BY.",
            "- 'Per store / per brand / per category / per year / per month' means GROUP BY that dimension.",
            "- For GROUP BY queries, always include the grouped dimension as the first SELECT column.",
            "- 'Placed in YEAR' or 'from YEAR' on orders means WHERE EXTRACT(YEAR FROM order_date) = YEAR.",
            "- When asked 'which X has the highest/best Y' (single winner), use LIMIT 1.",
            "- When showing products joined with other tables, always include p.list_price in SELECT.",
            "- When querying staffs without asking for store info, SELECT first_name, last_name, email — do not JOIN stores.",
            "- For 'show order items', select: order_id, order_date, product_name, quantity, list_price.",
        ]
        if few_shot_block:
            parts += [
                "",
                "=== EXAMPLES ===",
                few_shot_block,
                "",
                "Now generate a SQL query for the following question.",
            ]
        return "\n".join(parts)

    def _build_few_shot_block(self, prompt: str) -> str:
        if self.retriever is None:
            return ""
        examples = self.retriever.retrieve(prompt, k=6)
        lines = []
        for ex in examples:
            lines.append(f"Q: {ex['question']}")
            lines.append(f"SQL: {ex['sql']}")
            lines.append("")
        return "\n".join(lines).strip()

    def _extract_sql(self, text: str) -> str:
        """
        Strip markdown code fences and prose, returning only the SQL.
        Handles:
          ```sql ... ```
          ``` ... ```
          Inline leading SELECT / WITH / etc.
        """
        # Remove markdown code blocks
        for pattern in (r"```sql\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
            m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()

        # Walk lines until we find a SQL keyword, then keep everything after
        sql_starters = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE")
        lines = text.split("\n")
        sql_lines = []
        capturing = False
        for line in lines:
            if not capturing and line.strip().upper().startswith(sql_starters):
                capturing = True
            if capturing:
                sql_lines.append(line)
        if sql_lines:
            return "\n".join(sql_lines).strip()

        # Last resort: return the whole text stripped
        return text.strip()

    def _format_schema(self) -> str:
        """Format the DB schema as a readable string for the LLM prompt."""
        parts = []
        for table_name, columns in self.schema.items():
            cols = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
            parts.append(f"  {table_name}({cols})")
        return "\n".join(parts)
