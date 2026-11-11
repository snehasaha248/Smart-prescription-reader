"""
database.py
The pharmacy inventory, stored in a SQLite database file (data/pharmacy.db).

Tables:
  medicines     med_id, name, strength, form, stock, shelf
  dispense_log  every medicine a pharmacist approved and dispensed
"""
import sqlite3
from datetime import datetime

import pandas as pd

DB_PATH = "data/pharmacy.db"
CSV_PATH = "data/medicines.csv"


def connect():
    return sqlite3.connect(DB_PATH)


def init_db(reset=False):
    """Create the tables. Fill the medicines table from the CSV the first time (or on reset)."""
    with connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS dispense_log (
                            time TEXT, med_id INTEGER, name TEXT, strength TEXT,
                            quantity INTEGER, shelf TEXT, pharmacist TEXT)""")
        csv = pd.read_csv(CSV_PATH).fillna("")
        exists = conn.execute("SELECT name FROM sqlite_master WHERE name='medicines'").fetchone()
        if exists and not reset:
            # Reload if the CSV gained new medicines or columns since the database was made.
            current = pd.read_sql("SELECT * FROM medicines", conn)
            reset = list(current.columns) != list(csv.columns) or len(current) != len(csv)
        if reset or not exists:
            csv.to_sql("medicines", conn, if_exists="replace", index=False)
            if reset:
                conn.execute("DELETE FROM dispense_log")


def get_inventory():
    """All medicines as a Pandas table (DataFrame)."""
    with connect() as conn:
        return pd.read_sql("SELECT * FROM medicines ORDER BY med_id", conn)


def get_log():
    with connect() as conn:
        return pd.read_sql("SELECT * FROM dispense_log ORDER BY time DESC", conn)


def dispense(med_id, quantity, pharmacist):
    """Reduce stock after a pharmacist has verified the item, and record it in the log."""
    with connect() as conn:
        row = conn.execute("SELECT name, strength, stock, shelf FROM medicines WHERE med_id=?",
                           (med_id,)).fetchone()
        if row is None:
            raise ValueError(f"No medicine with id {med_id}")
        name, strength, stock, shelf = row
        if stock < quantity:
            raise ValueError(f"Only {stock} of {name} {strength} in stock")
        conn.execute("UPDATE medicines SET stock = stock - ? WHERE med_id=?", (quantity, med_id))
        conn.execute("INSERT INTO dispense_log VALUES (?,?,?,?,?,?,?)",
                     (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), med_id, name, strength,
                      quantity, shelf, pharmacist))


if __name__ == "__main__":
    # Quick test: python database.py
    init_db(reset=True)
    print(get_inventory().head(10))
    print("Total medicines:", len(get_inventory()))
