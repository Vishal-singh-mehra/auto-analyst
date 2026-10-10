from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, MetaData, Table
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from forecaster import _forecastable_numeric_cols
import pandas as pd

engine = create_engine("sqlite:///auto_analyst.db")
Session = sessionmaker(bind=engine)


def save_to_database(df, source_filename, table_name="sales_data"):
    """
    Append a cleaned dataframe into a SQLite table.
    Adds tracking columns so we know which upload each row came from.
    """
    df = df.copy()
    df["_source_file"] = source_filename
    df["_uploaded_at"] = datetime.now().isoformat()

    # if_exists="append": add rows to the table if it already exists,
    # create it fresh (inferring column types from the dataframe) if not.
    df.to_sql(table_name, engine, if_exists="append", index=False)

    return table_name


def load_from_database(table_name="sales_data"):
    """Read the whole table back out as a dataframe."""
    return pd.read_sql_table(table_name, engine)

def get_numeric_columns(table_name="sales_data"):
    df = load_from_database(table_name)
    return _forecastable_numeric_cols(df)


def get_all_tables():
    """List every table currently in the database."""
    from sqlalchemy import inspect
    inspector = inspect(engine)
    return inspector.get_table_names()