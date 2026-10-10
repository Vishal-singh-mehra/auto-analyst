import os
import pandas as pd

from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename

from cleaner import clean_dataframe
from database import save_to_database, load_from_database, get_numeric_columns
from forecaster import detect_forecastable_columns, run_forecast, forecast_for_column
from ydata_profiling import ProfileReport

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB cap


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("index.html", error="Please choose a file.")
    if not file.filename.lower().endswith(".csv"):
        return render_template("index.html", error="Only .csv files are supported.")

    filename = secure_filename(file.filename)
    raw_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(raw_path)

    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        return render_template("index.html", error=f"Could not read that CSV: {e}")

    cleaned_df, log, summary = clean_dataframe(df)

    clean_path = os.path.join(app.config["UPLOAD_FOLDER"], f"cleaned_{filename}")
    cleaned_df.to_csv(clean_path, index=False)

    save_to_database(cleaned_df, source_filename=filename)

    forecast_info = detect_forecastable_columns(cleaned_df)
    forecast_data = None
    forecast_message = None
    selection_reason = None
    fill_reason = None

    if forecast_info:
        forecast_data = run_forecast(
            cleaned_df,
            forecast_info["date_col"],
            forecast_info["value_col"],
            fill_strategy=forecast_info["fill_strategy"],
        )
        selection_reason = forecast_info["selection_reason"]
        fill_reason = forecast_info["fill_reason"]
        if forecast_data is None:
            forecast_message = "Not enough historical data to forecast reliably."
    else:
        forecast_message = "No suitable date + numeric column pair found — forecasting skipped."

    report_name = f"{filename.rsplit('.', 1)[0]}_report.html"
    profile = ProfileReport(cleaned_df, title=f"EDA: {filename}", minimal=True)
    profile.to_file(os.path.join("reports", report_name))

    preview = cleaned_df.head(10).to_html(classes="preview", index=False)

    return render_template(
        "result.html",
        filename=filename,
        log=log,
        summary=summary,
        preview=preview,
        columns=list(cleaned_df.columns),
        report_name=report_name,
        forecast_data=forecast_data,
        forecast_message=forecast_message,
        has_forecast=(forecast_data is not None),
        selection_reason=selection_reason,
        fill_reason=fill_reason,
    )

@app.route("/report/<path:name>")
def report(name):
    return send_from_directory("reports", name)

@app.route("/dashboard")
def dashboard():
    try:
        df = load_from_database()
    except Exception:
        return render_template(
            "dashboard.html",
            no_data=True,
            columns=[],
            default_col=None,
            suggestion_reason=None
        )

    numeric_cols = get_numeric_columns()

    if not numeric_cols:
        return render_template(
            "dashboard.html",
            no_data=True,
            columns=[],
            default_col=None,
            suggestion_reason=None
        )

    forecast_info = detect_forecastable_columns(df)

    default_col = (
        forecast_info["value_col"]
        if forecast_info
        else numeric_cols[0]
    )

    suggestion_reason = (
        forecast_info["selection_reason"]
        if forecast_info
        else None
    )

    return render_template(
        "dashboard.html",
        no_data=False,
        columns=numeric_cols,
        default_col=default_col,
        suggestion_reason=suggestion_reason,
    )

@app.route("/api/forecast")
def api_forecast():
    column = request.args.get("column")
    if not column:
        return jsonify({"error": "No column specified"}), 400

    df = load_from_database()
    if column not in df.columns:
        return jsonify({"error": f"Column '{column}' not found"}), 400

    result = forecast_for_column(df, column)
    if result is None:
        return jsonify({"error": "Not enough data to forecast this column"}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)