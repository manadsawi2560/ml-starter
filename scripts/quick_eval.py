# save as scripts/quick_eval.py and run: python scripts/quick_eval.py
import json
import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

THR_JSON = "artifacts/metrics.json"
MODEL = "artifacts/model.joblib"
DATA = "data/raw/processed.cleveland.data"
COLS = [
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

with open(THR_JSON) as f:
    info = json.load(f)
thr = info.get("threshold", 0.5)

df = pd.read_csv(DATA, header=None, names=COLS, na_values=["?"])
df["target"] = (df["num"].astype(float) > 0).astype(int)
X = df.drop(columns=["num", "target"])
y = df["target"].values

m = joblib.load(MODEL)
prob = m.predict_proba(X)[:, 1]
pred = (prob >= thr).astype(int)

tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(
    {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
)
