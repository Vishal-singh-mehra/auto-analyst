import os
import re
import threading
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from forecaster import COUNT_COL, _pick_date_column, date_columns, forecastable_metrics

engine = create_engine(os.environ.get("DATABASE_URL", "sqlite:///auto_analyst.db"))

SAMPLE_ROWS = 3000          # enough rows to learn column types without loading the whole table
LEGACY_TABLE = "sales_data"  # the single shared table used by earlier versions of this project

_lock = threading.Lock()
_sample_cache = {}           # table name -> {"version": ..., "df": ...}
_profile_cache = {}          # a larger sample, used only for the data overview
PROFILE_ROWS = int(os.environ.get("PROFILE_SAMPLE_ROWS", "20000"))


def table_name_for(filename):
    """Every uploaded file gets its OWN table, so different datasets can never mix or collide."""
    stem = filename.rsplit(".", 1)[0].lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:50] or "data"
    return f"ds_{slug}"


def _all_tables():
    return inspect(engine).get_table_names()


def get_all_tables():
    return _all_tables()


def _is_dataset_table(name):
    return name.startswith("ds_") or name == LEGACY_TABLE


def list_datasets():
    """All uploaded datasets, newest first: [{table, name, rows, uploaded_at}]."""
    found = []
    for table in _all_tables():
        if not _is_dataset_table(table):
            continue
        with engine.connect() as conn:
            rows, latest, source = conn.execute(
                text(f'SELECT COUNT(*), MAX(_uploaded_at), MAX(_source_file) FROM "{table}"')
            ).one()
        if not rows:
            continue
        name = "Earlier uploads (combined)" if table == LEGACY_TABLE else source
        found.append({"table": table, "name": name, "rows": rows, "uploaded_at": latest or ""})
    return sorted(found, key=lambda d: d["uploaded_at"], reverse=True)


def _reconvert_dates(df):
    """
    SQLite has no real DATE type, so a date column can come back as text.
    This re-detects any text column that is really dates.
    """
    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            converted = pd.to_datetime(df[col], errors="coerce", format="mixed")
            non_null = df[col].notna().sum()
            if non_null > 0 and converted.notna().sum() / non_null >= 0.8:
                df[col] = converted
    return df


def save_to_database(df, source_filename):
    """Save a cleaned dataset in its own table. Uploading the same file again replaces it."""
    table_name = table_name_for(source_filename)
    # assign() adds the two tracking columns without making a full second copy of a big table
    df = df.assign(_source_file=source_filename, _uploaded_at=datetime.now().isoformat())
    # chunksize keeps memory flat while saving big files
    df.to_sql(table_name, engine, if_exists="replace", index=False, chunksize=5000)
    with _lock:
        _sample_cache.pop(table_name, None)          # new data: the cached samples are out of date
        _profile_cache.pop(table_name, None)
    return table_name


def dataset_version(table_name):
    """A cheap fingerprint of one table. It changes whenever that dataset is uploaded again."""
    if table_name not in _all_tables():
        raise LookupError("No such dataset")
    with engine.connect() as conn:
        count, latest = conn.execute(
            text(f'SELECT COUNT(*), MAX(_uploaded_at) FROM "{table_name}"')
        ).one()
    if not count:
        raise LookupError("No data yet")
    return (count, latest)


def load_sample(table_name):
    """A small random sample of the table, kept in memory until new data arrives."""
    version = dataset_version(table_name)
    with _lock:
        cached = _sample_cache.get(table_name)
        if cached and cached["version"] == version:
            return cached["df"]

    # Picking random row ids first keeps this light even on a very large table.
    query = text(
        f'SELECT * FROM "{table_name}" WHERE rowid IN '
        f'(SELECT rowid FROM "{table_name}" ORDER BY RANDOM() LIMIT {SAMPLE_ROWS})'
    )
    sample = _reconvert_dates(pd.read_sql_query(query, engine))
    with _lock:
        _sample_cache[table_name] = {"version": version, "df": sample}
    return sample


def get_numeric_columns(table_name):
    """Everything that can be forecast for this dataset (including 'record_count' for event-style data)."""
    total_rows = dataset_version(table_name)[0]
    return forecastable_metrics(load_sample(table_name), total_rows)


def load_forecast_frame(value_col, table_name):
    """
    Load ONLY the columns a forecast needs (the timeline and the metric), not the whole table.
    Returns None if the table has no date column.
    """
    sample = load_sample(table_name)
    date_cols = date_columns(sample)
    if not date_cols:
        return None
    date_col = _pick_date_column(sample, date_cols)

    # All names come from the table's own columns (never straight from the browser), so quoting is safe.
    if value_col == COUNT_COL:
        query = f'SELECT "{date_col}" FROM "{table_name}"'
    else:
        if value_col not in sample.columns:
            raise KeyError(value_col)
        query = f'SELECT "{date_col}", "{value_col}" FROM "{table_name}"'
    df = pd.read_sql_query(text(query), engine)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    return df


def delete_dataset(table_name):
    """Remove one uploaded dataset. Only tables this app created can be removed."""
    if table_name not in {item["table"] for item in list_datasets()}:
        return False
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE "{table_name}"'))
    with _lock:
        _sample_cache.pop(table_name, None)
        _profile_cache.pop(table_name, None)
    return True


def load_profile_sample(table_name):
    """Up to PROFILE_ROWS random rows, for the data overview. Kept in memory until new data arrives."""
    version = dataset_version(table_name)
    with _lock:
        cached = _profile_cache.get(table_name)
        if cached and cached["version"] == version:
            return cached["df"]
    query = text(
        f'SELECT * FROM "{table_name}" WHERE rowid IN '
        f'(SELECT rowid FROM "{table_name}" ORDER BY RANDOM() LIMIT {PROFILE_ROWS})'
    )
    df = pd.read_sql_query(query, engine)
    # Only the columns the small sample showed to be dates are converted: parsing every text column is slow.
    for column in date_columns(load_sample(table_name)):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")
    with _lock:
        _profile_cache[table_name] = {"version": version, "df": df}
    return df


def table_date_range(table_name):
    """First and last date of the main date column, and how many days that spans (exact, from the whole table)."""
    sample = load_sample(table_name)
    cols = date_columns(sample)
    if not cols:
        return None
    column = _pick_date_column(sample, cols)
    with engine.connect() as conn:
        first, last = conn.execute(text(f'SELECT MIN("{column}"), MAX("{column}") FROM "{table_name}"')).one()
    first, last = pd.to_datetime(first, errors="coerce"), pd.to_datetime(last, errors="coerce")
    if pd.isna(first) or pd.isna(last):
        return None
    return {"column": column, "first": first.strftime("%Y-%m-%d"), "last": last.strftime("%Y-%m-%d"),
            "days": int((last.normalize() - first.normalize()).days + 1)}
