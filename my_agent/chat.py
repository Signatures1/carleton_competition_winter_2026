"""
SQL Query Writer Agent — Terminal Chat Interface

Ask questions in plain English; the agent generates and runs the SQL for you.

Usage
-----
    python chat.py              # uses settings from .env / secrets.env
    python chat.py --no-exec   # print the SQL only, do not execute it

Commands
--------
    quit / exit / q   Exit the session
    schema            Print all table names
    help              Show this help text
"""

import os
import sys
import argparse
import textwrap

import duckdb

# Load .env then secrets.env so credentials stay out of the main config file
try:
    from dotenv import load_dotenv
    _here = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_here, ".env"),        override=False)
    load_dotenv(os.path.join(_here, "secrets.env"), override=False)
except ImportError:
    pass

from db.bike_store import BikeStoreDb
from agent import QueryWriter

DB_PATH = "bike_store.db"

# ── ANSI colour helpers (disabled automatically on Windows without ANSI support)
_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("TERM") == "xterm"

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

BOLD  = lambda t: _c("1",    t)
DIM   = lambda t: _c("2",    t)
CYAN  = lambda t: _c("96",   t)
GREEN = lambda t: _c("92",   t)
YELLOW= lambda t: _c("93",   t)
RED   = lambda t: _c("91",   t)


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_db(db_path: str) -> None:
    if not os.path.exists(db_path):
        print(DIM("  Database not found — downloading from Kaggle …"))
        BikeStoreDb(db_path=db_path)
    else:
        print(DIM(f"  Database ready: {db_path}"))


def run_sql(sql: str, db_path: str):
    """Execute sql and return (rows, col_names) or raise on error."""
    con = duckdb.connect(database=db_path, read_only=True)
    try:
        cur = con.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return rows, cols
    finally:
        con.close()


def print_table(rows: list, cols: list, max_rows: int = 20) -> None:
    """Print query results as a plain-text table."""
    if not rows:
        print(DIM("  (no rows returned)"))
        return

    display = rows[:max_rows]
    # Compute column widths
    widths = [len(str(c)) for c in cols]
    for row in display:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    sep   = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    hdr   = "|" + "|".join(f" {str(c).ljust(w)} " for c, w in zip(cols, widths)) + "|"

    print(DIM(sep))
    print(BOLD(hdr))
    print(DIM(sep))
    for row in display:
        line = "|" + "|".join(f" {str(v).ljust(w)} " for v, w in zip(row, widths)) + "|"
        print(line)
    print(DIM(sep))

    if len(rows) > max_rows:
        print(DIM(f"  … {len(rows) - max_rows} more rows not shown"))
    print(DIM(f"  {len(rows)} row(s)"))


def print_help() -> None:
    print(textwrap.dedent("""
    Commands
    --------
      quit / exit / q   Exit the session
      schema            List all database tables
      help              Show this message

    Just type any plain-English question and the agent will generate SQL for it.
    Examples:
      How many customers are there?
      What are the top 5 most expensive products?
      Show orders placed in 2019
      Which store has the most inventory?
    """).strip())


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat for the SQL Query Writer Agent")
    parser.add_argument("--no-exec", action="store_true",
                        help="Print the generated SQL but do not execute it")
    parser.add_argument("--db", default=DB_PATH,
                        help=f"Path to DuckDB database (default: {DB_PATH})")
    args = parser.parse_args()

    db_path = args.db
    host    = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
    model   = os.getenv("OLLAMA_MODEL", "llama3.2")

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print(BOLD("=" * 62))
    print(BOLD("  SQL Query Writer Agent — Terminal Chat"))
    print(BOLD("=" * 62))
    print(f"  {DIM('Model :')} {model}")
    print(f"  {DIM('Server:')} {host}")
    print(f"  {DIM('DB    :')} {db_path}")
    if args.no_exec:
        print(f"  {YELLOW('[--no-exec: SQL will be shown but not run]')}")
    print(BOLD("=" * 62))
    print()

    # ── Initialise ────────────────────────────────────────────────────────────
    print(DIM("Loading database …"))
    ensure_db(db_path)

    print(DIM("Initialising agent …"))
    agent = QueryWriter(db_path=db_path)

    tables = list(agent.schema.keys())
    print(DIM(f"Ready. {len(tables)} tables loaded."))
    print(DIM('Type "help" for commands, "quit" to exit.'))
    print()

    # ── REPL ──────────────────────────────────────────────────────────────────
    while True:
        try:
            raw = input(CYAN("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(DIM("Goodbye!"))
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("quit", "exit", "q"):
            print(DIM("Goodbye!"))
            break

        if cmd == "help":
            print_help()
            continue

        if cmd == "schema":
            print()
            for t in tables:
                cols = ", ".join(c["name"] for c in agent.schema[t])
                print(f"  {BOLD(t)}({DIM(cols)})")
            print()
            continue

        # ── Generate SQL ──────────────────────────────────────────────────────
        print()
        print(DIM("  Generating SQL …"))
        try:
            sql = agent.generate_query(raw)
        except Exception as e:
            print(RED(f"  Agent error: {e}"))
            print()
            continue

        print()
        print(BOLD("Generated SQL"))
        print(BOLD("─" * 50))
        print(GREEN(sql))
        print(BOLD("─" * 50))

        if args.no_exec:
            print()
            continue

        # ── Execute ───────────────────────────────────────────────────────────
        print()
        print(DIM("  Running query …"))
        try:
            rows, cols = run_sql(sql, db_path)
        except Exception as e:
            print(RED(f"  Execution error: {e}"))
            print()
            continue

        print()
        print(BOLD("Results"))
        print_table(rows, cols)
        print()


if __name__ == "__main__":
    main()
