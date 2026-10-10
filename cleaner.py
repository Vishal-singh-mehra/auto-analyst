import pandas as pd

MISSING_THRESHOLD = 0.5   # drop columns more than 50% empty


def try_convert_numeric(series):
    """Turn text like '1,250' or ' 42 ' into real numbers, if most of it converts."""
    if series.dtype != "object":
        return None

    stripped = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    converted = pd.to_numeric(stripped, errors="coerce")

    # only accept if at least 80% of non-empty values converted successfully
    original_filled = series.notna().sum()

    if (
        original_filled > 0
        and converted.notna().sum() / original_filled >= 0.8
    ):
        return converted

    return None


def try_convert_datetime(series):
    """Turn text like '2024-03-01' into real dates, if most of it converts."""
    if series.dtype != "object":
        return None

    converted = pd.to_datetime(
        series,
        errors="coerce",
        format="mixed"
    )

    original_filled = series.notna().sum()

    if (
        original_filled > 0
        and converted.notna().sum() / original_filled >= 0.8
    ):
        return converted

    return None


def clean_dataframe(df):
    """Clean any dataframe. Returns (cleaned_df, log_of_actions)."""

    log = []
    rows_before, cols_before = df.shape

    # 1. Tidy up column names
    df.columns = [
        str(c).strip().replace(" ", "_").lower()
        for c in df.columns
    ]

    # 2. Drop mostly-empty columns
    for col in list(df.columns):

        missing_ratio = df[col].isna().mean()

        if missing_ratio > MISSING_THRESHOLD:

            df = df.drop(columns=[col])

            log.append(
                f"Dropped '{col}' — {missing_ratio:.0%} missing"
            )

    # 3. Drop constant columns
    for col in list(df.columns):

        if df[col].nunique(dropna=True) <= 1:

            df = df.drop(columns=[col])

            log.append(
                f"Dropped '{col}' — only one unique value"
            )

    # 4. Drop duplicate rows
    dupes = df.duplicated().sum()

    if dupes > 0:

        df = df.drop_duplicates()

        log.append(
            f"Removed {dupes} duplicate rows"
        )

    # 5. Fix data types
    for col in df.columns:

        # Try datetime first
        as_date = try_convert_datetime(df[col])

        if as_date is not None:

            df[col] = as_date

            log.append(
                f"Converted '{col}' to datetime"
            )

            continue

        # Then try numeric
        as_num = try_convert_numeric(df[col])

        if as_num is not None:

            df[col] = as_num

            log.append(
                f"Converted '{col}' to numeric"
            )

    # 6. Standardize text values
    for col in df.columns:

        if pd.api.types.is_object_dtype(df[col]):

            # Count unique non-empty values
            unique_count = df[col].nunique(dropna=True)

            # Standardize columns that look categorical
            if unique_count <= 50:

                df[col] = (
                    df[col]
                    .astype("string")
                    .str.strip()
                    .str.title()
                )

                log.append(
                    f"Standardized text formatting in '{col}'"
                )

    # 7. Fill in what's missing
    for col in df.columns:

        if df[col].isna().sum() == 0:
            continue

        # Numeric → median
        if pd.api.types.is_numeric_dtype(df[col]):

            value = df[col].median()

            df[col] = df[col].fillna(value)

            log.append(
                f"Filled '{col}' missing values "
                f"with median ({value:.2f})"
            )

        # Datetime → drop rows with missing dates
        elif pd.api.types.is_datetime64_any_dtype(df[col]):

            missing_dates = df[col].isna().sum()

            df = df.dropna(subset=[col])

            log.append(
                f"Dropped {missing_dates} rows with "
                f"missing dates in '{col}'"
            )

        # Text → mode
        else:

            modes = df[col].mode()

            value = modes[0] if not modes.empty else "Unknown"

            df[col] = df[col].fillna(value)

            log.append(
                f"Filled '{col}' missing values "
                f"with mode ('{value}')"
            )

    # 8. Create summary
    summary = {
        "rows_before": rows_before,
        "cols_before": cols_before,
        "rows_after": df.shape[0],
        "cols_after": df.shape[1],
    }

    if not log:

        log.append(
            "No issues found — data was already clean."
        )

    return df, log, summary