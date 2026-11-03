import pandas as pd
import pytest

from docqa.errors import DocQAError
from docqa.tables import Filter, TableQuery, describe_tables, run_table_query

KEY = "sales.csv::csv"


@pytest.fixture
def tables():
    frame = pd.DataFrame(
        {
            "region": ["East", "East", "West", "West", "North"],
            "product": ["Laptop", "Mouse", "Laptop", "Monitor", "Mouse"],
            "amount": [1200, 50, 1800, 600, 25],
        }
    )
    return {KEY: frame}


def _query(**kwargs):
    return TableQuery(table=KEY, **kwargs)


def test_sum_with_filter(tables):
    q = _query(operation="sum", column="amount",
               filters=[Filter(column="region", op="==", value="East")])
    result = run_table_query(tables, q)
    assert "1250" in result
    assert "region == 'East'" in result


def test_count_rows_after_filter(tables):
    q = _query(operation="count", filters=[Filter(column="product", op="==", value="Laptop")])
    assert run_table_query(tables, q).endswith("= 2")


def test_mean_min_max(tables):
    assert run_table_query(tables, _query(operation="mean", column="amount")).endswith("= 735")
    assert run_table_query(tables, _query(operation="min", column="amount")).endswith("= 25")
    assert run_table_query(tables, _query(operation="max", column="amount")).endswith("= 1800")


def test_numeric_comparison_filter(tables):
    q = _query(operation="count", filters=[Filter(column="amount", op=">", value="100")])
    assert run_table_query(tables, q).endswith("= 3")


def test_contains_is_case_insensitive(tables):
    q = _query(operation="count", filters=[Filter(column="product", op="contains", value="lap")])
    assert run_table_query(tables, q).endswith("= 2")


def test_group_by_sum(tables):
    result = run_table_query(tables, _query(operation="sum", column="amount", group_by="region"))
    lines = result.splitlines()
    assert "East: 1250" in lines
    assert "West: 2400" in lines
    assert "North: 25" in lines


def test_unknown_column_lists_valid_columns(tables):
    with pytest.raises(DocQAError) as exc:
        run_table_query(tables, _query(operation="sum", column="nope"))
    message = str(exc.value)
    assert "nope" in message and "region" in message and "amount" in message


def test_unknown_table_lists_valid_tables(tables):
    with pytest.raises(DocQAError) as exc:
        run_table_query(tables, TableQuery(table="missing", operation="count"))
    assert KEY in str(exc.value)


def test_sum_on_text_column_raises(tables):
    with pytest.raises(DocQAError):
        run_table_query(tables, _query(operation="sum", column="region"))


def test_list_respects_limit():
    frame = pd.DataFrame({"n": range(100)})
    result = run_table_query({KEY: frame}, _query(operation="list", column="n", limit=500))
    assert len(result.splitlines()) <= 51  # one heading line plus at most 50 rows


def test_filter_value_is_never_executed(tables, tmp_path):
    marker = tmp_path / "pwned"
    hostile = f"__import__('os').system('touch {marker}')"
    q = _query(operation="count", filters=[Filter(column="region", op="==", value=hostile)])
    assert run_table_query(tables, q).endswith("= 0")
    assert not marker.exists()


def test_describe_tables_names_every_column(tables):
    description = describe_tables(tables)
    assert KEY in description
    for name in ("region", "product", "amount"):
        assert name in description
    assert "5 rows" in description


def test_small_decimals_are_not_rounded_away():
    t = {KEY: pd.DataFrame({"rate": [0.004, 0.0035, 0.012]})}
    assert run_table_query(t, _query(operation="min", column="rate")).endswith("= 0.0035")
    assert run_table_query(t, _query(operation="mean", column="rate")).endswith("= 0.0065")


def test_all_missing_column_reports_no_values():
    t = {KEY: pd.DataFrame({"x": [float("nan")] * 3})}
    for operation in ("sum", "mean", "min", "max"):
        result = run_table_query(t, _query(operation=operation, column="x"))
        assert result.endswith("= no values"), result


def test_group_by_output_is_capped():
    from docqa.tables import MAX_LIST_ROWS

    frame = pd.DataFrame({"id": [f"id{i}" for i in range(80)], "amount": list(range(80))})
    result = run_table_query({KEY: frame}, _query(operation="sum", column="amount", group_by="id"))
    assert len(result.splitlines()) <= MAX_LIST_ROWS + 3
    assert "30 more groups" in result
    assert "id79: 79" in result  # the largest groups are the ones kept
