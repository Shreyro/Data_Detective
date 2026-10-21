"""SQL guardrails. Every query - whether written by an LLM or by rules - passes through here.

Layers:
  1. Parse with sqlglot (must be exactly one statement, SELECT/WITH only)
  2. AST allow-list: only the approved table, no DDL/DML, no file/system functions
  3. Row cap: inject LIMIT when missing
  4. Bind check: DuckDB EXPLAIN catches unknown columns / type errors before execution
The connection itself is also read-only with external access disabled (see db.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import duckdb
import sqlglot
from sqlglot import exp

from .config import MAX_ROWS

ALLOWED_TABLES = {"marketing_performance"}
FORBIDDEN_NODES = tuple(
    getattr(exp, n) for n in
    ("Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command", "Copy", "Merge",
     "Set", "Use", "Pragma", "Attach", "Detach", "Transaction", "Commit", "Rollback", "Install", "Export")
    if hasattr(exp, n)
)
BLOCKED_FUNC_PREFIXES = ("read_", "duckdb_", "pragma_", "glob", "getenv", "query", "sniff_", "load", "parquet_", "json_")


@dataclass
class Validation:
    ok: bool
    sql: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def clean(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sql, flags=re.I).strip()
    return sql.rstrip(";").strip()


def validate(sql: str, warehouse=None, max_rows: int = MAX_ROWS) -> Validation:
    sql = clean(sql)
    errors: list[str] = []
    warnings: list[str] = []
    if not sql:
        return Validation(False, sql, ["Empty query."])
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as e:
        return Validation(False, sql, [f"Syntax error: {str(e).splitlines()[0]}"])
    if len(statements) != 1 or statements[0] is None:
        return Validation(False, sql, ["Exactly one statement is allowed."])
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return Validation(False, sql, [f"Only SELECT queries are allowed (got {type(tree).__name__})."])
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            errors.append(f"Forbidden operation: {type(node).__name__}.")
        if isinstance(node, exp.Anonymous) and node.name.lower().startswith(BLOCKED_FUNC_PREFIXES):
            errors.append(f"Function not allowed: {node.name}.")

    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):
            errors.append("Table functions are not allowed.")
        elif t.name.lower() not in ctes and t.name.lower() not in ALLOWED_TABLES:
            errors.append(f"Table `{t.name}` is not allowed. Use: {', '.join(sorted(ALLOWED_TABLES))}.")
    if errors:
        return Validation(False, sql, sorted(set(errors)))

    if any(isinstance(s, exp.Star) for s in tree.find_all(exp.Star)) and not tree.find(exp.Count):
        warnings.append("SELECT * returns every column; prefer explicit columns.")
    if not tree.args.get("limit"):
        tree = tree.limit(max_rows) if isinstance(tree, exp.Select) else sqlglot.select("*").from_(tree.subquery("q")).limit(max_rows)
        warnings.append(f"No LIMIT found; added LIMIT {max_rows}.")
    final = tree.sql(dialect="duckdb", pretty=True)

    if warehouse is not None:
        try:
            warehouse.explain(final)
        except duckdb.Error as e:
            return Validation(False, final, [f"Binder error: {str(e).splitlines()[0]}"], warnings)
    return Validation(True, final, [], warnings)
