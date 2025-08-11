from pathlib import Path
import pandas as pd

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def load_uci_cleveland(path: str, na_values=("?",)) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, header=None, names=COLUMNS, na_values=list(na_values))
    # target: num > 0 => 1 else 0
    df["target"] = (df["num"].astype(float) > 0).astype(int)
    df = df.drop(columns=["num"])
    return df
