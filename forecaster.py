import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

TARGET_KEYWORDS = [
    "sales", "revenue", "amount", "total", "profit",
    "quantity", "orders", "count", "units", "volume",
]

ZERO_FILL_KEYWORDS = TARGET_KEYWORDS

# Columns that are identifiers/codes, not measurements — never candidates to forecast
IGNORE_KEYWORDS = [
    "_id", "id_", "postal_code", "zipcode", "zip_code", "pincode", "_code", "code_",
]


def _is_id_like(col_name):
    name = col_name.lower()
    if name == "id":
        return True
    return any(kw in name for kw in IGNORE_KEYWORDS)


def _pick_date_column(df, date_cols):
    """
    Pick whichever datetime column spans the most real calendar days —
    NOT whichever has the most raw unique values. A time-only column
    (e.g. '16:37:47') converts to datetime but all lands on one fake day,
    so raw uniqueness would wrongly favor it over a real multi-year date column.
    """
    return max(date_cols, key=lambda c: df[c].dt.normalize().nunique())


def _forecastable_numeric_cols(df):
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith("_")
        and not _is_id_like(c)
    ]


def detect_forecastable_columns(df):
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric_cols = _forecastable_numeric_cols(df)

    if not date_cols or not numeric_cols:
        return None

    date_col = _pick_date_column(df, date_cols)
    if df[date_col].dt.normalize().nunique() < 10:
        return None

    name_match = next(
        (c for c in numeric_cols if any(kw in c.lower() for kw in TARGET_KEYWORDS)),
        None,
    )

    if name_match:
        value_col = name_match
        selection_reason = f"Selected '{value_col}' — column name matches a common metric term."
    else:
        def safe_std(c):
            s = df[c].std()
            return s if s == s else 0

        value_col = max(numeric_cols, key=safe_std)
        selection_reason = (
            f"No column name matched common metrics (sales, revenue, etc.) — "
            f"guessed '{value_col}' based on highest variation. Verify this is the right column."
        )

    if any(kw in value_col.lower() for kw in ZERO_FILL_KEYWORDS):
        fill_strategy = "zero"
        fill_reason = f"Missing dates filled with 0 — treating gaps in '{value_col}' as no recorded activity."
    else:
        fill_strategy = "ffill"
        fill_reason = f"Missing dates filled by carrying forward the last value — '{value_col}' doesn't look like a sum/count, so 0 would likely be wrong."

    return {
        "date_col": date_col,
        "value_col": value_col,
        "selection_reason": selection_reason,
        "fill_strategy": fill_strategy,
        "fill_reason": fill_reason,
    }


def run_forecast(df, date_col, value_col, fill_strategy="zero", periods=30, min_points=10):
    series = (
        df[[date_col, value_col]]
        .dropna()
        .groupby(date_col)[value_col]
        .sum()
        .sort_index()
    )

    if len(series) < 2:
        return None

    series = series.asfreq("D")
    series = series.fillna(0) if fill_strategy == "zero" else series.ffill()

    if len(series) < min_points:
        return None

    try:
        model = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
        forecast = model.forecast(periods)
    except Exception:
        return None

    return {
        "history_dates": series.index.strftime("%Y-%m-%d").tolist(),
        "history_values": series.values.tolist(),
        "forecast_dates": forecast.index.strftime("%Y-%m-%d").tolist(),
        "forecast_values": forecast.values.tolist(),
    }


def forecast_for_column(df, value_col, periods=30):
    """Forecast a specific, user-chosen column — used by the dashboard dropdown."""
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not date_cols:
        return None
    date_col = _pick_date_column(df, date_cols)  # <-- now uses the SAME fixed logic

    fill_strategy = "zero" if any(kw in value_col.lower() for kw in ZERO_FILL_KEYWORDS) else "ffill"
    return run_forecast(df, date_col, value_col, fill_strategy=fill_strategy, periods=periods)