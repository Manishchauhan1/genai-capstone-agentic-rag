"""A safe table-query tool for the Analyst agent (FR-8).

The LLM never writes code. It fills in a TableQuery (data), and this module runs
it with plain pandas operations. Nothing here uses eval, exec or DataFrame.query.
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel

from docqa.errors import DocQAError

MAX_LIST_ROWS = 50


class Filter(BaseModel):
    column: str
    op: Literal["==", "!=", ">", "<", ">=", "<=", "contains"]
    value: str | float


class TableQuery(BaseModel):
    table: str
    operation: Literal["count", "sum", "mean", "min", "max", "list"]
    column: str | None = None
    filters: list[Filter] = []
    group_by: str | None = None
    limit: int = 20


def describe_tables(tables: dict[str, pd.DataFrame]) -> str:
    """One line per table: key, row count and `column (dtype)` list."""
    lines = []
    for key, frame in tables.items():
        columns = ", ".join(f"{name} ({dtype})" for name, dtype in frame.dtypes.items())
        lines.append(f"{key}: {len(frame)} rows; columns: {columns}")
    return "\n".join(lines)


def _require_column(frame: pd.DataFrame, name: str) -> None:
    if name not in frame.columns:
        valid = ", ".join(str(column) for column in frame.columns)
        raise DocQAError(f"There is no column {name!r}. Valid columns: {valid}.")


def _filter_mask(frame: pd.DataFrame, condition: Filter) -> pd.Series:
    _require_column(frame, condition.column)
    series = frame[condition.column]
    if condition.op == "contains":
        return series.astype(str).str.contains(str(condition.value), case=False, regex=False, na=False)
    if pd.api.types.is_numeric_dtype(series):
        try:
            value = float(condition.value)
        except (TypeError, ValueError):
            raise DocQAError(
                f"Column {condition.column!r} is numeric, so {condition.value!r} is not a valid value."
            ) from None
        compare = {
            "==": series == value, "!=": series != value, ">": series > value,
            "<": series < value, ">=": series >= value, "<=": series <= value,
        }
        return compare[condition.op].fillna(False)
    if condition.op not in ("==", "!="):
        raise DocQAError(
            f"Column {condition.column!r} is text, so only ==, != and contains work on it."
        )
    same = series.astype(str).str.strip().str.casefold() == str(condition.value).strip().casefold()
    return same if condition.op == "==" else ~same


def _format(value) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        # Keep small values (0.0035) intact; round only large ones to 2 places.
        return str(round(value, 2)) if abs(value) >= 1 else f"{value:.6g}"
    return str(value)


def _capped_lines(series, header: str) -> str:
    """One `group: value` line per group; large results keep only the biggest groups."""
    total = len(series)
    if total > MAX_LIST_ROWS:
        series = series.sort_values(ascending=False, kind="stable").head(MAX_LIST_ROWS)
    lines = [f"{group}: {_format(value)}" for group, value in series.items()]
    if total > MAX_LIST_ROWS:
        lines.append(f"... and {total - MAX_LIST_ROWS} more groups ({total} groups in all; "
                     f"the {MAX_LIST_ROWS} largest are shown)")
    return header + "\n".join(lines)


def _where(filters: list[Filter]) -> str:
    if not filters:
        return ""
    parts = [f"{f.column} {f.op} {f.value!r}" for f in filters]
    return " where " + " and ".join(parts)


def resolve_table(tables: dict[str, pd.DataFrame], name: str) -> str:
    """Find the table key a model meant. Accepts the exact key, the same key in any
    letter case, or a file name that has exactly one table."""
    if name in tables:
        return name
    wanted = name.strip().casefold()
    exact = [key for key in tables if key.casefold() == wanted]
    by_file = [key for key in tables if key.split("::")[0].casefold() == wanted]
    for matches in (exact, by_file):
        if len(matches) == 1:
            return matches[0]
    raise DocQAError(f"There is no table {name!r}. Valid tables: {', '.join(tables)}.")


def run_table_query(tables: dict[str, pd.DataFrame], query: TableQuery) -> str:
    """Run one query and return a short, quotable description of the result."""
    frame = tables[resolve_table(tables, query.table)]
    for condition in query.filters:
        frame = frame[_filter_mask(frame, condition)]
    where = _where(query.filters)

    if query.column is not None:
        _require_column(frame, query.column)
    if query.group_by is not None:
        _require_column(frame, query.group_by)

    if query.operation == "list":
        limit = min(max(query.limit, 1), MAX_LIST_ROWS)
        shown = frame[[query.column]] if query.column else frame
        rows = ["; ".join(f"{c}: {_format(v)}" for c, v in zip(shown.columns, row))
                for row in shown.head(limit).itertuples(index=False, name=None)]
        what = query.column or "all columns"
        return f"first {len(rows)} rows of {what} in {query.table}{where}:\n" + "\n".join(rows)

    if query.operation == "count":
        if query.group_by:
            counts = frame.groupby(query.group_by).size()
            return _capped_lines(counts, f"count of rows in {query.table}{where} by {query.group_by}:\n")
        return f"count of rows in {query.table}{where} = {len(frame)}"

    if query.column is None:
        raise DocQAError(f"The {query.operation} operation needs a column.")
    if not pd.api.types.is_numeric_dtype(frame[query.column]):
        raise DocQAError(f"Column {query.column!r} is text, so {query.operation} cannot be used on it.")

    if query.group_by:
        grouped = frame.groupby(query.group_by)[query.column].agg(query.operation)
        header = f"{query.operation} of {query.column} in {query.table}{where} by {query.group_by}:\n"
        return _capped_lines(grouped, header)

    label = f"{query.operation} of {query.column} in {query.table}{where}"
    if frame.empty:
        return f"{label} = no matching rows"
    if frame[query.column].notna().sum() == 0:
        return f"{label} = no values"
    value = frame[query.column].agg(query.operation)
    return f"{label} = {_format(value)}"
