import csv
import gc
import os
import shutil
import threading

os.environ.setdefault("TQDM_DISABLE", "1")      # keep the profiling library's progress bars out of the server log

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from cleaner import clean_dataframe
from database import (
    dataset_version, delete_dataset, get_numeric_columns, list_datasets, load_forecast_frame, load_sample, save_to_database,
)
from forecaster import COUNT_COL, ForecastError, date_columns, detect_forecastable_columns, forecast_for_column

app = Flask(__name__)

# ---- settings (can be changed with environment variables, no code edits needed) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)      # Git doesn't keep empty folders, so create them here
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024
ENABLE_EDA = os.environ.get("ENABLE_EDA", "1") != "0"           # set to 0 on very small servers
EDA_MAX_ROWS = int(os.environ.get("EDA_MAX_ROWS", "20000"))     # the report uses a sample above this size
# "auto" = the full report (correlations, missing-value view, interactions, data sample) for small datasets and
# a quicker version for big ones. Use "minimal" on a very small server, or "full" to always get everything.
EDA_MODE = os.environ.get("EDA_MODE", "auto")
FULL_EDA_MAX_CELLS = int(os.environ.get("FULL_EDA_MAX_CELLS", "30000"))   # rows x columns

# ---- forecast cache: the same column is not re-forecast until new data is uploaded ----
_forecast_cache = {}
_cache_lock = threading.Lock()
CACHE_LIMIT = 20


def cached_forecast(dataset, column):
    version = dataset_version(dataset)
    key = (dataset, column, version)
    with _cache_lock:
        if key in _forecast_cache:
            return _forecast_cache[key]

    df = load_forecast_frame(column, dataset)
    if df is None:
        raise ForecastError("This dataset has no date column, so there is nothing to forecast over.")
    result = forecast_for_column(df, column)          # may raise ForecastError with a plain-language reason

    with _cache_lock:
        for old_key in [k for k in _forecast_cache if k[0] == dataset and k[2] != version]:
            del _forecast_cache[old_key]                          # that dataset changed: drop its stale results
        if result is not None:
            if len(_forecast_cache) >= CACHE_LIMIT:
                _forecast_cache.pop(next(iter(_forecast_cache)))
            _forecast_cache[key] = result
    return result


def resolve_dataset(requested):
    """Use the dataset asked for if it exists, otherwise the most recently uploaded one."""
    datasets = list_datasets()
    tables = {d["table"] for d in datasets}
    if requested in tables:
        return requested, datasets
    return (datasets[0]["table"] if datasets else None), datasets


def summarize_forecast(result, column):
    """The headline of the forecast, in plain numbers, for the results page."""
    n = len(result["forecast_dates"])
    is_sum = result["aggregation"] == "sum"
    now = result["forecast_total"] if is_sum else result["forecast_average"]
    last = sum(result["history_values"][-n:])
    before = last if is_sum else last / n
    accuracy = result.get("accuracy") or {}
    return {
        "metric": "number of records per day" if column == COUNT_COL else column.replace("_", " "),
        "days": n,
        "kind": "total" if is_sum else "average per day",
        "value": now,
        "change_pct": None if before == 0 else (now - before) / abs(before) * 100,
        "method": accuracy.get("method_label"),
        "weekly_error_pct": accuracy.get("weekly_error_pct"),
        "total_error_pct": accuracy.get("total_error_pct"),
        "weekly_word": "total" if is_sum else "average",
    }


def read_csv_flexibly(path):
    """
    Open CSV files saved by Excel in other regions: a different text encoding (cp1252) and ';' or tab
    as the separator instead of a comma.
    """
    last_error = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, encoding=encoding, newline="") as handle:
                head = handle.read(8192)
            try:
                separator = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
            except csv.Error:
                separator = ","
            return pd.read_csv(path, encoding=encoding, sep=separator)
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error


def make_eda_report(df, filename):
    """Build the profiling report. Returns (report_name or None, note for the user)."""
    if not ENABLE_EDA:
        return None, "The detailed data report is switched off on this server."
    try:
        # Imported here, not at the top: this library is large, so the server stays light until it is needed.
        from ydata_profiling import ProfileReport

        data, notes = df, []
        if len(df) > EDA_MAX_ROWS:
            data = df.sample(EDA_MAX_ROWS, random_state=0)
            notes.append(f"The report uses a random sample of {EDA_MAX_ROWS:,} of {len(df):,} rows.")
        cells = data.shape[0] * data.shape[1]
        full = EDA_MODE == "full" or (EDA_MODE == "auto" and cells <= FULL_EDA_MAX_CELLS)
        if not full and EDA_MODE != "minimal":
            notes.append("This is the quick version of the report (no correlations or interaction charts) because the file is large.")
        report_name = f"{filename.rsplit('.', 1)[0]}_report.html"
        ProfileReport(data, title=f"EDA: {filename}", minimal=not full, progress_bar=False).to_file(
            os.path.join(REPORT_FOLDER, report_name)
        )
        note = " ".join(notes) or None
        return report_name, note
    except Exception:
        app.logger.exception("EDA report failed")
        return None, "The detailed data report could not be generated for this file."


DEMO_FILE = os.path.join(BASE_DIR, "static", "demo_daily_sales_2023_2025.csv")


def home_page(error=None):
    """The upload page, with the datasets uploaded so far."""
    datasets = []
    try:
        for d in list_datasets():
            datasets.append({**d, "when": (d["uploaded_at"] or "")[:16].replace("T", " ")})
    except Exception:
        app.logger.exception("could not list datasets")
    return render_template("index.html", datasets=datasets, has_demo=os.path.exists(DEMO_FILE), error=error)


@app.route("/")
def home():
    return home_page()


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        return home_page(error="Please choose a file.")
    if not file.filename.lower().endswith(".csv"):
        return home_page(error="Only .csv files are supported.")

    filename = secure_filename(file.filename)
    raw_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(raw_path)
    return analyze_file(raw_path, filename)


@app.route("/demo", methods=["POST"])
def demo():
    """One click: run the built-in sample file through exactly the same steps as an upload."""
    if not os.path.exists(DEMO_FILE):
        return home_page(error="The demo file is not installed.")
    filename = os.path.basename(DEMO_FILE)
    raw_path = os.path.join(UPLOAD_FOLDER, filename)
    shutil.copyfile(DEMO_FILE, raw_path)
    return analyze_file(raw_path, filename)


def analyze_file(raw_path, filename):
    try:
        df = read_csv_flexibly(raw_path)
    except Exception as e:
        return home_page(error=f"Could not read that CSV: {e}")

    cleaned_df, log, summary = clean_dataframe(df)
    del df
    gc.collect()

    cleaned_df.to_csv(os.path.join(UPLOAD_FOLDER, f"cleaned_{filename}"), index=False)
    preview = cleaned_df.head(10).to_html(classes="preview", index=False)

    report_name, report_note = make_eda_report(cleaned_df, filename)
    table_name = save_to_database(cleaned_df, source_filename=filename)

    forecast_info = detect_forecastable_columns(cleaned_df, total_rows=len(cleaned_df))
    forecast_data = None
    forecast_summary = None
    forecast_message = None
    selection_reason = None
    fill_reason = None

    if forecast_info:
        selection_reason = forecast_info["selection_reason"]
        fill_reason = forecast_info["fill_reason"]
        try:
            forecast_data = forecast_for_column(cleaned_df, forecast_info["value_col"])
            if forecast_data is None:
                forecast_message = "Not enough historical data to forecast reliably."
            else:
                forecast_summary = summarize_forecast(forecast_data, forecast_info["value_col"])
        except ForecastError as problem:
            forecast_message = str(problem)
    else:
        forecast_message = "No suitable date + numeric column pair found — forecasting skipped."

    return render_template(
        "result.html",
        filename=filename,
        log=log,
        summary=summary,
        preview=preview,
        columns=list(cleaned_df.columns),
        report_name=report_name,
        report_note=report_note,
        forecast_message=forecast_message,
        has_forecast=(forecast_data is not None),
        forecast_summary=forecast_summary,
        selection_reason=selection_reason,
        fill_reason=fill_reason,
        dataset=table_name,
    )


@app.route("/download/<name>")
def download_cleaned(name):
    """The cleaned version of an uploaded file."""
    safe = secure_filename(name)
    path = os.path.join(UPLOAD_FOLDER, f"cleaned_{safe}")
    if not os.path.exists(path):
        return "That cleaned file is not available. Upload the file again.", 404
    return send_from_directory(UPLOAD_FOLDER, f"cleaned_{safe}", as_attachment=True, download_name=f"cleaned_{safe}")


@app.route("/dataset/delete", methods=["POST"])
def dataset_delete():
    delete_dataset(request.form.get("table", ""))
    with _cache_lock:
        _forecast_cache.clear()
    return redirect("/")


@app.route("/report/<path:name>")
def report(name):
    return send_from_directory(REPORT_FOLDER, name)


@app.route("/dashboard")
def dashboard():
    dataset, datasets = resolve_dataset(request.args.get("dataset"))
    empty = dict(no_data=True, message=None, columns=[], default_col=None, suggestion_reason=None,
                 datasets=datasets, dataset=dataset)
    if dataset is None:
        return render_template("dashboard.html", **empty)

    try:
        sample = load_sample(dataset)
        total_rows = dataset_version(dataset)[0]
        metrics = get_numeric_columns(dataset)
    except Exception:
        app.logger.exception("dashboard failed")
        return render_template("dashboard.html", **empty)

    if not metrics:
        has_date = bool(date_columns(sample))
        empty["message"] = ("This dataset has no numeric column to forecast." if has_date
                            else "This dataset has no date column, so there is nothing to forecast over.")
        return render_template("dashboard.html", **empty)

    forecast_info = detect_forecastable_columns(sample, total_rows=total_rows)
    default_col = forecast_info["value_col"] if forecast_info else metrics[0]
    suggestion_reason = forecast_info["selection_reason"] if forecast_info else None

    return render_template(
        "dashboard.html",
        no_data=False,
        message=None,
        columns=metrics,
        default_col=default_col,
        suggestion_reason=suggestion_reason,
        datasets=datasets,
        dataset=dataset,
    )


@app.route("/api/forecast")
def api_forecast():
    column = request.args.get("column")
    if not column:
        return jsonify({"error": "No column specified"}), 400

    dataset, _ = resolve_dataset(request.args.get("dataset"))
    if dataset is None:
        return jsonify({"error": "No data yet — upload a CSV first."}), 400

    try:
        allowed = get_numeric_columns(dataset)
    except Exception:
        return jsonify({"error": "Could not read this dataset."}), 400
    if not allowed:
        return jsonify({"error": "This dataset has no date column, so there is nothing to forecast over."}), 400
    if column not in allowed:                      # also keeps anything unexpected out of the database query
        return jsonify({"error": f"'{column}' is not available for forecasting in this dataset"}), 400

    try:
        result = cached_forecast(dataset, column)
    except ForecastError as problem:
        return jsonify({"error": str(problem)}), 400
    if result is None:
        return jsonify({"error": "Not enough data to forecast this column"}), 400
    return jsonify(result)


if __name__ == "__main__":
    # Debug mode lets anyone who can reach the page run code on your computer, so it is OFF unless you ask.
    # For development with auto-reload, set FLASK_DEBUG=1 first (see the instructions).
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
    )
