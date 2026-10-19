import pandas as pd

MISSING_THRESHOLD = 0.5   # drop columns more than 50% empty


def _is_text_column(series):
    """True for text-ish columns, across pandas versions (object in <3.0, str in 3.0+)."""
    return series.dtype == "object" or pd.api.types.is_string_dtype(series)


def _sample_looks_like(series, convert):
    """
    Cheap pre-check: try the conversion on up to 200 random values first.
    Converting a whole column only to find out it was ordinary text is what makes big files slow.
    """
    filled = series.dropna()
    if filled.empty:
        return False
    sample = filled.sample(min(200, len(filled)), random_state=0)
    converted = convert(sample)
    return converted.notna().sum() / len(sample) >= 0.8


_NOISE = r"(?i)[₹$€£¥%\s\u00a0]|inr|rs\.?|usd|eur|gbp"


def _to_numeric(series):
    """Turn text like '₹1,23,456.50', 'Rs. 450', '12%' or '(300)' into numbers."""
    text = series.astype(str).str.strip()
    negative = text.str.match(r"^\(.*\)$")                      # (300) means -300 in accounting files
    text = text.str.replace(_NOISE, "", regex=True)
    with_comma = text.str.contains(",", regex=False)
    # European style: 1.234,50 means 1234.50 (comma = decimal point). Indian/US style: 1,234.50 or 1,23,456.
    # A comma followed by only 1-2 digits at the very end is the decimal marker; thousands groups have 3.
    if with_comma.any() and text[with_comma].str.contains(r",\d{1,2}$").mean() > 0.8:
        text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        text = text.str.replace(",", "", regex=False)
    text = text.str.strip("()")
    numbers = pd.to_numeric(text, errors="coerce")
    return numbers.where(~negative, -numbers)


_SLASH_DATE = r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\s*$"


def _date_order(series):
    """
    Dates like 05/01/2023 are ambiguous. Look at all the values: a first number above 12
    proves day-first (13/01/2023), a second number above 12 proves month-first (01/13/2023).
    If nothing proves it, assume day-first (the usual order in India and most of the world).
    Returns "day", "month", or None when the values are not written like d/m/y at all.
    """
    parts = series.dropna().astype(str).str.extract(_SLASH_DATE).dropna()
    if len(parts) < max(1, int(series.notna().sum() * 0.8)):
        return None
    first, second = parts[0].astype(int), parts[1].astype(int)
    if (second > 12).any() and not (first > 12).any():
        return "month"
    return "day"


def _to_datetime(series):
    order = _date_order(series)
    return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=(order == "day"))


def try_convert_numeric(series):
    if not _is_text_column(series) or not _sample_looks_like(series, _to_numeric):
        return None
    converted = _to_numeric(series)
    original_filled = series.notna().sum()
    if original_filled > 0 and converted.notna().sum() / original_filled >= 0.8:
        return converted
    return None


def try_convert_datetime(series):
    if not _is_text_column(series) or not _sample_looks_like(series, _to_datetime):
        return None
    converted = _to_datetime(series)
    original_filled = series.notna().sum()
    if original_filled > 0 and converted.notna().sum() / original_filled >= 0.8:
        return converted
    return None


def clean_dataframe(df):
    log = []
    rows_before, cols_before = df.shape
    missing_before = int(df.isna().sum().sum())
    df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
    for col in list(df.columns):
        missing_ratio = df[col].isna().mean()
        if missing_ratio > MISSING_THRESHOLD:
            df = df.drop(columns=[col])
            log.append(f"Dropped '{col}' — {missing_ratio:.0%} missing")
    for col in list(df.columns):
        if df[col].nunique(dropna=True) <= 1:
            df = df.drop(columns=[col])
            log.append(f"Dropped '{col}' — only one unique value")
    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates()
        log.append(f"Removed {dupes} duplicate rows")
    for col in df.columns:
        as_date = try_convert_datetime(df[col])
        if as_date is not None:
            order = _date_order(df[col])
            df[col] = as_date
            log.append(f"Converted '{col}' to datetime")
            if order == "day":
                log.append(f"Read '{col}' dates as day/month/year (e.g. 05/01/2023 = 5 January 2023)")
            elif order == "month":
                log.append(f"Read '{col}' dates as month/day/year (e.g. 01/13/2023 = 13 January 2023)")
            continue
        as_num = try_convert_numeric(df[col])
        if as_num is not None:
            df[col] = as_num
            log.append(f"Converted '{col}' to numeric")
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            value = df[col].median()
            df[col] = df[col].fillna(value)
            log.append(f"Filled '{col}' missing values with median ({value:.2f})")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df = df.dropna(subset=[col])
            log.append(f"Dropped rows with missing dates in '{col}'")
        else:
            modes = df[col].mode()
            value = modes[0] if not modes.empty else "Unknown"
            df[col] = df[col].fillna(value)
            log.append(f"Filled '{col}' missing values with mode ('{value}')")
    summary = {"rows_before": rows_before, "cols_before": cols_before,
               "rows_after": df.shape[0], "cols_after": df.shape[1],
               "duplicates_removed": int(dupes),
               "missing_before": missing_before, "missing_after": int(df.isna().sum().sum())}
    if not log:
        log.append("No issues found — data was already clean.")
    return df, log, summary