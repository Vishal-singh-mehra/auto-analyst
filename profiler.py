"""Plain-language data overview: what is in the dataset, how its columns are spread, and what moves together."""
import numpy as np
import pandas as pd

from forecaster import TARGET_KEYWORDS, _is_id_like


def _clean(value):
    """Turn numpy / pandas values into plain JSON-friendly ones (NaN becomes None)."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def visible_columns(df):
    return [c for c in df.columns if not str(c).startswith("_")]


def column_kind(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "number"
    return "text"


def summarize_columns(sample):
    """One line per column: its type, how many different values, and a typical value or range."""
    rows, kinds = [], {"number": 0, "date": 0, "text": 0}
    for name in visible_columns(sample):
        series = sample[name]
        kind = column_kind(series)
        kinds[kind] += 1
        info = {"name": name, "kind": kind, "distinct": int(series.nunique(dropna=True))}
        if kind == "number":
            info.update(min=_clean(series.min()), median=_clean(series.median()), mean=_clean(series.mean()),
                        max=_clean(series.max()), id_like=bool(_is_id_like(name)))
        elif kind == "date":
            if series.dropna().dt.normalize().nunique() <= 1:        # only a time of day, e.g. 16:37:47
                info.update(time_only=True, min=series.min().strftime("%H:%M"), max=series.max().strftime("%H:%M"))
            else:
                info.update(min=_clean(series.min()), max=_clean(series.max()))
        else:
            counts = series.value_counts(dropna=True)
            if len(counts):
                info.update(top=str(counts.index[0]), top_share=float(counts.iloc[0] / counts.sum()))
        rows.append(info)
    return rows, kinds


def correlations(sample, max_columns=12):
    """How strongly the number columns rise and fall together, plus the strongest pairs."""
    numbers = [c for c in visible_columns(sample)
               if column_kind(sample[c]) == "number" and not _is_id_like(c) and sample[c].nunique() > 1]
    # Business-looking columns first (sales, profit...), then the rest in file order.
    numbers = sorted(numbers, key=lambda c: 0 if any(k in c.lower() for k in TARGET_KEYWORDS) else 1)[:max_columns]
    if len(numbers) < 2:
        return {"columns": numbers, "matrix": [], "pairs": []}
    corr = sample[numbers].corr(numeric_only=True)
    matrix = [[_clean(round(float(corr.iloc[i, j]), 2)) for j in range(len(numbers))] for i in range(len(numbers))]
    pairs = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            r = corr.iloc[i, j]
            if np.isfinite(r):
                pairs.append({"a": numbers[i], "b": numbers[j], "r": round(float(r), 2)})
    pairs.sort(key=lambda p: -abs(p["r"]))
    return {"columns": numbers, "matrix": matrix, "pairs": pairs[:6]}


def distribution(sample, column, bins=20, top=10):
    """The data for one chart: how the values of a single column are spread."""
    series = sample[column]
    kind = column_kind(series)
    if kind == "number":
        values = series.dropna().to_numpy(dtype=float)
        if len(values) == 0:
            return {"column": column, "kind": "empty"}
        distinct = np.unique(values)
        if len(distinct) <= 15:                                   # a handful of values (1-5 ratings...): one bar each
            counts = pd.Series(values).value_counts().sort_index()
            return {"column": column, "kind": "bars", "labels": [_clean(v) for v in counts.index],
                    "counts": [int(c) for c in counts.values]}
        counts, edges = np.histogram(values, bins=bins)
        return {"column": column, "kind": "histogram", "edges": [_clean(e) for e in edges], "counts": [int(c) for c in counts]}
    if kind == "date" and series.dropna().dt.normalize().nunique() <= 1:
        hours = series.dropna().dt.hour.value_counts().reindex(range(24), fill_value=0)
        return {"column": column, "kind": "hours", "labels": [f"{h:02d}:00" for h in hours.index], "counts": [int(c) for c in hours.values]}
    if kind == "date":
        months = series.dropna().dt.to_period("M").value_counts().sort_index().tail(60)
        return {"column": column, "kind": "months", "labels": [str(p) for p in months.index], "counts": [int(c) for c in months.values]}
    counts = series.dropna().astype(str).value_counts()
    shown = counts.head(top)
    return {"column": column, "kind": "top", "labels": list(shown.index), "counts": [int(c) for c in shown.values],
            "distinct": int(len(counts)), "shown_share": float(shown.sum() / counts.sum()) if counts.sum() else 0.0}


def build_overview(sample, total_rows, date_info):
    columns, kinds = summarize_columns(sample)
    return {
        "rows": int(total_rows),
        "columns": len(columns),
        "kinds": kinds,
        "sample_rows": int(len(sample)),
        "sampled": bool(len(sample) < total_rows),
        "date": date_info,
        "column_table": columns,
        "correlations": correlations(sample),
    }
