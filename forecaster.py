import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Column names that strongly suggest "this is the metric worth forecasting"
TARGET_KEYWORDS = [
    "sales", "revenue", "amount", "total", "profit",
    "quantity", "orders", "count", "units", "volume",
    "views", "visits", "clicks", "sessions", "tickets",
    "transactions", "downloads", "passengers",
]

# Metrics that are ADDED UP over a day (a day's sales is the sum of its orders; a day with no rows had none).
# Anything else, such as a temperature, a rating or a price, is AVERAGED over the day instead.
ADDITIVE_KEYWORDS = TARGET_KEYWORDS + ["cost", "spend", "expense", "income", "items", "calls", "signups", "installs"]
ZERO_FILL_KEYWORDS = ADDITIVE_KEYWORDS

# A built-in metric for event-style data (one row per ticket, visit, order...): how many rows per day.
COUNT_COL = "record_count"


class ForecastError(ValueError):
    """The data can't be forecast, for a reason worth telling the user in plain words."""


def is_additive(col_name):
    return any(kw in col_name.lower() for kw in ADDITIVE_KEYWORDS)


# Columns that are identifiers/codes, not measurements — never candidates to forecast
IGNORE_KEYWORDS = [
    "_id", "id_", "postal_code", "zipcode", "zip_code", "pincode", "_code", "code_",
]


def _is_id_like(col_name):
    name = col_name.lower()
    if name == "id":
        return True
    return any(kw in name for kw in IGNORE_KEYWORDS)


def date_columns(df):
    """Real date columns. The app's own tracking columns (names starting with '_') never count."""
    return [c for c in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[c]) and not str(c).startswith("_")]


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


def _rows_per_day(df, date_col, total_rows=None):
    days = df[date_col].dt.normalize()
    span = (days.max() - days.min()).days + 1
    return (total_rows or len(df)) / max(span, 1)


def forecastable_metrics(df, total_rows=None):
    """
    Everything the user may forecast. For event-style data (several rows per day) the number of
    records per day is offered too. `total_rows` lets callers pass the true table size when `df` is a sample.
    """
    date_cols = date_columns(df)
    if not date_cols:
        return []
    date_col = _pick_date_column(df, date_cols)
    metrics = _forecastable_numeric_cols(df)
    if _rows_per_day(df, date_col, total_rows) >= 1.5:
        metrics = [COUNT_COL] + metrics
    return metrics


def detect_forecastable_columns(df, total_rows=None):
    date_cols = date_columns(df)
    if not date_cols:
        return None

    date_col = _pick_date_column(df, date_cols)
    if df[date_col].dt.normalize().nunique() < 10:
        return None

    metrics = forecastable_metrics(df, total_rows)
    if not metrics:
        return None
    numeric_cols = [c for c in metrics if c != COUNT_COL]

    name_match = next(
        (c for c in numeric_cols if any(kw in c.lower() for kw in TARGET_KEYWORDS)),
        None,
    )

    if name_match:
        value_col = name_match
        selection_reason = f"Selected '{value_col}' — column name matches a common metric term."
    elif COUNT_COL in metrics:
        value_col = COUNT_COL
        per_day = _rows_per_day(df, date_col, total_rows)
        selection_reason = (
            f"Each row looks like one event (about {per_day:.0f} per day), so the number of records per day "
            f"is the most natural thing to forecast. Pick another column if you prefer."
        )
    else:
        def safe_std(c):
            s = df[c].std()
            return s if s == s else 0

        value_col = max(numeric_cols, key=safe_std)
        selection_reason = (
            f"No column name matched common metrics (sales, revenue, etc.) — "
            f"guessed '{value_col}' based on highest variation. Verify this is the right column."
        )

    if is_additive(value_col):
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


# ---------------------------------------------------------------------------
# Forecasting methods. Each takes (history, steps) and returns `steps` daily values.
# ---------------------------------------------------------------------------

def _future_index(train, steps):
    return pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=steps)


def _method_trend(train, steps):
    """Recent level and direction only."""
    train = train.iloc[-730:]
    return ExponentialSmoothing(train, trend="add", seasonal=None).fit().forecast(steps)


def _method_weekday(train, steps):
    """Recent level, direction and a day-of-the-week pattern."""
    train = train.iloc[-730:]
    return ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7).fit().forecast(steps)


def _method_last_year(train, steps):
    """Same period last year (smoothed), scaled by how much the business has grown since."""
    if steps > 364:
        raise ValueError("horizon too long for this method")
    smooth = train.rolling(7, center=True, min_periods=1).mean()
    index = _future_index(train, steps)
    base = smooth.reindex(index - pd.Timedelta(days=364)).values
    if np.isnan(base).any():
        raise ValueError("not enough history")
    recent, earlier = train.iloc[-91:].sum(), train.iloc[-455:-364].sum()
    growth = float(np.clip(recent / earlier, 0.5, 2.0)) if earlier > 0 else 1.0
    return pd.Series(base * growth, index=index)


def _method_yearly(train, steps):
    """Learns the yearly cycle from weekly totals (needs about two years of history)."""
    n = (len(train) // 7) * 7
    weekly = train.iloc[len(train) - n:].values.reshape(-1, 7).sum(axis=1)
    index = pd.date_range(end=train.index[-1], periods=len(weekly), freq="7D")
    fit = ExponentialSmoothing(pd.Series(weekly, index=index), trend=None,
                               seasonal="add", seasonal_periods=52).fit()
    weeks = int(np.ceil(steps / 7))
    per_day = fit.forecast(weeks).values / 7
    # Glide smoothly between the weekly levels instead of jumping in steps.
    daily = np.interp(np.arange(steps), np.arange(weeks) * 7 + 3, per_day)
    return pd.Series(daily, index=_future_index(train, steps))


# (key, function, days of history needed, plain-English label)
METHODS = [
    ("trend", _method_trend, 14, "recent level and trend"),
    ("weekday", _method_weekday, 28, "recent trend plus a day-of-the-week pattern"),
    ("last_year", _method_last_year, 455, "same period last year, scaled by recent growth"),
    ("yearly", _method_yearly, 735, "yearly cycle learned from past years"),
]
METHOD_BY_KEY = {m[0]: m for m in METHODS}


def _predict(method_fn, train, steps, floor):
    predicted = method_fn(train, steps)
    values = np.asarray(predicted, dtype=float)
    if np.isnan(values).any():
        raise ValueError("method produced no value")
    if floor is not None:
        values = np.maximum(values, floor)
    return pd.Series(values, index=_future_index(train, steps))


def _score_method(series, method_fn, origins, window, floor):
    """Test one method on the past. Returns None if it fails on any window."""
    day_errors, week_errors, total_errors, residuals, week_residuals = [], [], [], [], []
    weeks = window // 7
    for start, end in origins:
        train, test = series.iloc[:start], series.iloc[start:end]
        try:
            predicted = _predict(method_fn, train, window, floor)
        except Exception:
            return None
        miss = test.values - predicted.values
        residuals.extend(miss.tolist())
        if test.mean() > 0:
            day_errors.append(np.mean(np.abs(miss)) / test.mean() * 100)
        if test.sum() > 0:
            total_errors.append(abs(predicted.sum() - test.sum()) / test.sum() * 100)
        if weeks >= 1:
            actual_w = test.values[:weeks * 7].reshape(weeks, 7).sum(axis=1)
            predicted_w = predicted.values[:weeks * 7].reshape(weeks, 7).sum(axis=1)
            week_residuals.extend((actual_w - predicted_w).tolist())
            if actual_w.mean() > 0:
                week_errors.append(np.mean(np.abs(actual_w - predicted_w)) / actual_w.mean() * 100)
    if not day_errors or not week_errors:
        return None
    return {
        "daily_error_pct": float(np.mean(day_errors)),
        "weekly_error_pct": float(np.mean(week_errors)),
        "total_error_pct": float(np.mean(total_errors)) if total_errors else None,
        "residuals": residuals,
        "week_residuals": week_residuals,
        "windows": len(day_errors),
    }


def _choose_method(series, periods, max_origins=6, min_train=28):
    """
    Test every suitable method on the past (hide recent days, forecast them,
    compare with what really happened, repeat on several earlier windows) and
    keep whichever was most accurate. Returns None if there is too little history.
    """
    n = len(series)
    window = min(periods, n - min_train)
    if window < 7:
        return None

    origins = []
    for i in range(max_origins):
        end = n - i * window
        start = end - window
        if start < min_train:
            break
        origins.append((start, end))
    if not origins:
        return None

    floor = 0.0 if series.min() >= 0 else None
    contenders = [m for m in METHODS if m[2] <= n]

    # Every method must be judged on the same windows, so each needs enough history before the oldest one.
    while True:
        need = max(m[2] for m in contenders)
        common = [o for o in origins if o[0] >= need]
        if len(common) >= 3 or len(contenders) == 1:
            break
        contenders = [m for m in contenders if m[2] != need]
    if not common:
        common = origins

    scored = []
    for key, fn, _, label in contenders:
        score = _score_method(series, fn, common, window, floor)
        if score:
            scored.append((key, label, score))
    if not scored:
        return None

    key, label, best = min(scored, key=lambda t: (t[2]["daily_error_pct"] + t[2]["weekly_error_pct"]) / 2)
    low, high = np.percentile(best["residuals"], [5, 95])
    w_low, w_high = np.percentile(best["week_residuals"], [5, 95])
    return {
        "method": key,
        "method_label": label,
        "methods_tested": len(scored),
        "window_days": int(window),
        "windows": best["windows"],
        "daily_error_pct": round(best["daily_error_pct"], 1),
        "weekly_error_pct": round(best["weekly_error_pct"], 1),
        "total_error_pct": round(best["total_error_pct"], 1) if best["total_error_pct"] is not None else None,
        "band_low": float(low), "band_high": float(high),
        "week_band_low": float(w_low), "week_band_high": float(w_high),
    }


def _weekly_view(series, forecast, test, weeks_back=26, aggregate="sum"):
    """Weekly figures are far easier to read than 365 noisy daily points."""
    n = len(series)
    full_weeks = n // 7
    forecast_weeks = len(forecast) // 7
    if full_weeks < 8 or forecast_weeks < 1:
        return None

    # A week is the TOTAL of its 7 days for additive metrics (sales) and their AVERAGE otherwise (temperature).
    combine = (lambda blocks: blocks.sum(axis=1)) if aggregate == "sum" else (lambda blocks: blocks.mean(axis=1))
    scale = 1 if aggregate == "sum" else 7

    # Count 7-day blocks backwards from the last day of history, so no block is a partial week.
    tail = series.iloc[n - full_weeks * 7:]
    weekly = combine(tail.values.reshape(full_weeks, 7))[-weeks_back:]
    starts = tail.index[::7][-weeks_back:]

    f_weekly = combine(forecast.values[:forecast_weeks * 7].reshape(forecast_weeks, 7))
    f_starts = forecast.index[::7][:forecast_weeks]

    low = high = None
    if test:
        low = np.maximum(f_weekly + test["week_band_low"] / scale, 0).tolist()
        high = (f_weekly + test["week_band_high"] / scale).tolist()

    return {
        "history_dates": starts.strftime("%Y-%m-%d").tolist(),
        "history_values": weekly.tolist(),
        "forecast_dates": f_starts.strftime("%Y-%m-%d").tolist(),
        "forecast_values": f_weekly.tolist(),
        "forecast_low": low,
        "forecast_high": high,
    }


def _monthly_view(series, aggregate="sum"):
    """Whole-year picture: one point per full calendar month."""
    resampled = series.resample("MS")
    monthly = resampled.sum() if aggregate == "sum" else resampled.mean()
    if series.index[0].day != 1:
        monthly = monthly.iloc[1:]
    if series.index[-1] != series.index[-1] + pd.offsets.MonthEnd(0):
        monthly = monthly.iloc[:-1]
    if len(monthly) < 6:
        return None
    return {
        "dates": monthly.index.strftime("%Y-%m-%d").tolist(),
        "values": monthly.values.tolist(),
    }


def run_forecast(df, date_col, value_col, fill_strategy="zero", periods=30, min_points=10, aggregate=None):
    if value_col == COUNT_COL and COUNT_COL not in df.columns:
        df = df.assign(**{COUNT_COL: 1})                  # one record = 1, so a day's total is its row count
    if aggregate is None:
        aggregate = "sum" if fill_strategy == "zero" else "mean"

    # Group by CALENDAR DAY. Timestamps such as 2024-03-01 14:23:11 must all land on 2024-03-01.
    frame = pd.DataFrame({"day": df[date_col].dt.normalize().values, "value": df[value_col].values}).dropna()
    grouped = frame.groupby("day")["value"]
    series = (grouped.sum() if aggregate == "sum" else grouped.mean()).sort_index()
    rows_per_day = float(grouped.size().mean()) if len(series) else 1.0

    if len(series) < 2:
        return None

    gap = series.index.to_series().diff().dropna().dt.days.median()
    if gap > 3:
        raise ForecastError(
            f"This data has about one reading every {gap:.0f} days (weekly or monthly). "
            f"Forecasting needs daily data, and weekly or monthly data is not supported yet."
        )

    series = series.asfreq("D")
    series = series.fillna(0) if fill_strategy == "zero" else series.ffill()

    if len(series) < min_points:
        return None

    floor = 0.0 if series.min() >= 0 else None

    # Accuracy testing and method choice are extras: if they fail, a plain forecast still works.
    try:
        test = _choose_method(series, periods)
    except Exception:
        test = None

    forecast = None
    if test:
        try:
            forecast = _predict(METHOD_BY_KEY[test["method"]][1], series, periods, floor)
        except Exception:
            test = None
    if forecast is None:
        try:
            forecast = _predict(_method_trend, series, periods, floor)
        except Exception:
            return None

    result = {
        "history_dates": series.index.strftime("%Y-%m-%d").tolist(),
        "history_values": series.values.tolist(),
        "forecast_dates": forecast.index.strftime("%Y-%m-%d").tolist(),
        "forecast_values": forecast.values.tolist(),
        "forecast_total": float(forecast.sum()),
        "forecast_average": float(forecast.mean()),
        "aggregation": aggregate,
        "rows_per_day": round(rows_per_day, 1),
        "forecast_low": None,
        "forecast_high": None,
        "accuracy": None,
    }

    if test:
        low = forecast.values + test["band_low"]
        high = forecast.values + test["band_high"]
        if floor is not None:
            low = np.maximum(low, floor)
        result["forecast_low"] = low.tolist()
        result["forecast_high"] = high.tolist()
        result["accuracy"] = {k: v for k, v in test.items()
                              if k not in ("band_low", "band_high", "week_band_low", "week_band_high")}

    result["weekly"] = _weekly_view(series, forecast, test, aggregate=aggregate)
    result["monthly"] = _monthly_view(series, aggregate=aggregate)
    return result


def forecast_for_column(df, value_col, periods=30):
    """Forecast a specific, user-chosen column — used by the dashboard dropdown."""
    date_cols = date_columns(df)
    if not date_cols:
        raise ForecastError("This dataset has no date column, so there is nothing to forecast over.")
    date_col = _pick_date_column(df, date_cols)

    additive = is_additive(value_col)
    return run_forecast(
        df, date_col, value_col,
        fill_strategy="zero" if additive else "ffill",
        aggregate="sum" if additive else "mean",
        periods=periods,
    )