# src/ml_starter/models/train.py
import json
import joblib
from pathlib import Path
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline

from ml_starter.data.loader import load_uci_cleveland
from ml_starter.features.pipeline import build_preprocess

RAW_PATH = "data/raw/processed.cleveland.data"


def load_cfg(path="configs/train.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metrics(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    return {
        "recall": float(recall_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        # เติมได้ตามต้องการ เช่น precision, f1, specificity
    }


def main(cfg):
    # 1) Load data
    df = load_uci_cleveland(RAW_PATH)
    X, y = df.drop(columns=["target"]), df["target"].values

    # 2) Split train/val/test
    test_val = cfg["split"]["test_size"] + cfg["split"]["val_size"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X,
        y,
        test_size=test_val,
        random_state=cfg["seed"],
        stratify=y if cfg["split"].get("stratify", True) else None,
    )
    rel = cfg["split"]["val_size"] / test_val
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp,
        y_tmp,
        test_size=1 - rel,
        random_state=cfg["seed"],
        stratify=y_tmp if cfg["split"].get("stratify", True) else None,
    )

    # 3) Build pipeline + fit
    pre = build_preprocess()
    clf = LogisticRegression(**cfg["model"]["params"])
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_tr, y_tr)

    # 4) Predict prob
    val_prob = pipe.predict_proba(X_val)[:, 1]
    te_prob = pipe.predict_proba(X_te)[:, 1]

    # 5) หา threshold แบบ F1-optimal บนชุด validation
    P, R, T = precision_recall_curve(y_val, val_prob)  # T มีขนาด len(P)-1
    F1 = (2 * P * R) / (P + R + 1e-9)
    idx = int(np.nanargmax(F1))
    # ถ้า idx ชี้ไปจุดสุดท้าย (ที่ไม่มี threshold ใน T) ให้ใช้ 1.0
    thr_f1 = float(T[idx]) if idx < len(T) else 1.0

    # 6) ประเมินผลด้วย threshold F1-optimal
    val_metrics = metrics(y_val, val_prob, thr_f1)
    test_metrics = metrics(y_te, te_prob, thr_f1)

    # 7) Save artifacts
    Path(cfg["paths"]["artifacts"]).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, cfg["paths"]["model"])
    out = {
        "threshold_method": "f1_optimal",
        "threshold": thr_f1,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(Path(cfg["paths"]["artifacts"]) / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", cfg["paths"]["model"])
    print(out)


if __name__ == "__main__":
    main(load_cfg())
