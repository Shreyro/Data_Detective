"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DB_PATH = Path(os.getenv("DD_DB_PATH", ROOT / "data" / "marketing.duckdb"))
MODEL = os.getenv("DD_MODEL", "claude-sonnet-5-5")
FORCE_OFFLINE = os.getenv("DD_OFFLINE", "0") == "1"

MAX_ROWS = 1000          # hard cap on rows returned by any query
MAX_STEPS = 3            # max analysis queries per question
MAX_REPAIRS = 2          # SQL repair attempts per step
QUERY_TIMEOUT_S = 10     # kill queries that run longer than this
MAX_QUESTION_CHARS = 500
