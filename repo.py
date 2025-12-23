import pandas as pd
from pathlib import Path

def get_customer_row(customers_path: Path, customer_id: str):
    df = pd.read_csv(customers_path)
    row = df[df["customer_id"] == customer_id]
    return row
