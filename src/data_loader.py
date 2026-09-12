from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_openml

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

def load_data() -> pd.DataFrame:
    # OpenML dataset: public credit-card fraud classification data.
    dataset = fetch_openml(name="creditcard", version=1, as_frame=True, parser="auto")
    df = dataset.frame.copy()

    # OpenML may represent the target as strings/categories.
    if "Class" not in df.columns:
        target_name = dataset.target.name if hasattr(dataset.target, "name") else None
        if target_name and target_name in df.columns:
            df = df.rename(columns={target_name: "Class"})

    df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(int)

    return df
