import pytest

from data_detective.validator import validate


@pytest.mark.parametrize("sql", [
    "DROP TABLE campaigns",
    "DELETE FROM marketing_performance",
    "INSERT INTO marketing_performance VALUES (1)",
    "SELECT 1; SELECT 2",
    "SELECT * FROM campaigns",                                    # table not on allow-list
    "SELECT * FROM daily_performance",
    "SELECT * FROM read_csv('/etc/passwd')",                      # table function
    "SELECT getenv('HOME')",
    "COPY marketing_performance TO '/tmp/x.csv'",
    "ATTACH 'x.db' AS x",
    "SELEC channel FROM marketing_performance",                   # syntax
    "",
])
def test_rejects_unsafe_or_invalid(sql, wh):
    v = validate(sql, wh)
    assert not v.ok and v.errors


def test_unknown_column_caught_by_binder(wh):
    v = validate("SELECT nope FROM marketing_performance", wh)
    assert not v.ok and "Binder error" in v.errors[0]


def test_accepts_select_and_adds_limit(wh):
    v = validate("```sql\nSELECT channel, SUM(revenue) AS r FROM marketing_performance GROUP BY channel;\n```", wh)
    assert v.ok and "LIMIT 1000" in v.sql


def test_cte_allowed(wh):
    v = validate("WITH a AS (SELECT channel, revenue FROM marketing_performance) SELECT channel, SUM(revenue) FROM a GROUP BY 1", wh)
    assert v.ok


def test_connection_is_read_only(wh):
    with pytest.raises(Exception):
        wh.run("DELETE FROM daily_performance")
    with pytest.raises(Exception):
        wh.run("SELECT * FROM read_csv('/etc/passwd')")
