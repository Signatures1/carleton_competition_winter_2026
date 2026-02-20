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

class ExampleRetriever:
    """
    Given a pool of labeled (question, sql) examples, retrieves the k most
    semantically relevant examples for a new query using TF-IDF cosine
    similarity.

    This is the 'supervised learning' component: the TF-IDF vectorizer is
    *fitted* on the training split, and all downstream retrievals are made
    relative to that learned vocabulary.
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
            stop_words="english",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(questions)

    def retrieve(self, query: str, k: int = 4) -> list:
        """Return the k most similar examples to *query*."""
        query_vec = self.vectorizer.transform([query])
        sims = self._cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_k = self._np.argsort(sims)[::-1][:k]
        return [self.examples[i] for i in top_k]


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
            "- Use proper table aliases in JOIN queries.",
            "- Use DuckDB-compatible syntax.",
            "- For year/month extraction use EXTRACT(YEAR FROM col) / EXTRACT(MONTH FROM col).",
            "- For revenue calculations use: quantity * list_price * (1 - discount).",
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
        examples = self.retriever.retrieve(prompt, k=4)
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
