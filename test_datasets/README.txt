Auto-Analyst test datasets: upload each one and compare with what you should see.
These files are synthetic (made up), built to be awkward on purpose.

A_messy_retail.csv     dd/mm/yyyy dates, rupee amounts like "₹6,201.63", blank cells, 5 duplicate rows, headers with spaces.
                       Should: forecast "total_amount_(inr)" (not units), read 05/01/2023 as 5 January, remove the duplicates.
B_monthly_revenue.csv  one row per month.
                       Should: say it is monthly data and decline to forecast (not supported yet).
C_hourly_traffic.csv   hourly timestamps, 120 days.
                       Should: forecast page_views per day (total 712,120); avg_session_minutes is averaged (about 4.5), not added up.
D_daily_temperature.csv  daily temperature. Should: weekly AVERAGES (about 17-33), not sums.
E_students_no_date.csv   no date column. Should: say there is nothing to forecast over.
F_tickets_events.csv   one row per support ticket (4,000 rows).
                       Should: offer "Number of records per day" and use it by default.
G_weekly_orders.csv    one row per week. Should: decline, like B.
H_european_excel.csv   semicolon-separated, cp1252 encoding, numbers like "1.234,50", dates like 05.01.2023.
                       Should: load and forecast "umsatz" (total 960,919).
