# Auto-Analyst

Upload a sales CSV and Auto-Analyst cleans it, builds a data report, and forecasts the next 30 days. It also shows how far off the forecast has usually been on your own past data.

Built for small businesses that keep sales records but have no analyst, as a project for **UN Sustainable Development Goal 8** (Decent Work and Economic Growth), supporting the growth of small and medium-sized enterprises.

## What it does

- **Cleans messy files automatically.** Fixes text that is really a date or a number (including `dd/mm/yyyy` dates, `₹1,23,456.50` and European `1.234,50` numbers), removes duplicates, fills gaps, and lists every change it made.
- **Reads Excel exports.** Handles `;` separators and non-UTF-8 files.
- **Builds a data report.** A full report (distributions, missing values, correlations) for small files and a quicker one for large files.
- **Forecasts 30 days ahead.** It tests four methods on your own past data and uses the most accurate one.
- **Shows its reasoning.** Explains why a column was chosen, lets you pick another, and shows typical error with a shaded likely range.
- **Works on different kinds of data.** Adds up sales-type columns and averages things like temperature. For event-style data (one row per ticket or visit) it can forecast records per day.
- **Dashboard.** Weekly, daily and monthly views, light and dark themes, and downloads for the chart (PNG) and the forecast (CSV).

## How accurate is it?

Measured on a 138,116-row e-commerce dataset, on 14 past periods the forecast had never seen:

| | Typical error |
|---|---|
| A single day | about 14% |
| A week's total | about 6% |
| The 30-day total | about 4% |

Single days are mostly random noise, so the dashboard defaults to the weekly view. These figures describe how the method behaved on past data. They are not a guarantee, and sudden events (a sale, a new competitor) cannot be forecast. The yearly pattern is only used when there are about two years of history.

## Run it on your computer

Needs Python 3.10 to 3.13 (3.12 recommended).

```bash
git clone https://github.com/Vishal-singh-mehra/auto-analyst.git
cd auto-analyst
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and click **Try with demo data**.

The `test_datasets/` folder has eight deliberately awkward files (messy rupee amounts, timestamps, monthly data, no date column, European Excel format and more). Its `README.txt` says what each one should do.

## How it works

1. **Clean** (`cleaner.py`): detects dates and numbers hidden in text, removes duplicates and empty columns, fills gaps.
2. **Store** (`database.py`): each file gets its own table in a local SQLite database.
3. **Forecast** (`forecaster.py`): picks the date and metric columns, tries four methods (recent trend, day-of-week pattern, same period last year, yearly cycle), tests each on past data, and keeps the best.
4. **Show** (`app.py`, `templates/`, `static/`): Flask serves the pages and a JSON API; Chart.js draws the charts.

## Settings

Set as environment variables, for example in PowerShell: `$env:EDA_MODE="minimal"`.

| Variable | Default | What it does |
|---|---|---|
| `FLASK_DEBUG` | off | Set to `1` for auto-reload while developing. Keep off on a public server. |
| `MAX_UPLOAD_MB` | 50 | Largest file accepted. |
| `ENABLE_EDA` | 1 | Set to `0` to skip the data report. |
| `EDA_MODE` | auto | `auto`, `minimal` or `full`. Use `minimal` on a small server. |
| `EDA_MAX_ROWS` | 20000 | The report uses a sample above this many rows. |
| `DATABASE_URL` | local SQLite file | Where data is stored. |

## Project structure

```
app.py            routes and page logic
cleaner.py        data cleaning
database.py       storage (one table per uploaded file)
forecaster.py     column detection, forecasting methods, accuracy testing
templates/        HTML pages
static/           stylesheet and the demo file
test_datasets/    eight test files with expected results
requirements.txt  tested package versions
```

## Limitations

- Daily data only. Weekly or monthly files are declined with an explanation.
- CSV files only (save Excel files as CSV first).
- If no column name hints at what to forecast, it guesses and says so. You can always choose another.
- Tested on the included datasets, not yet on real shop data.

## Author

Vishal Singh Mehra
