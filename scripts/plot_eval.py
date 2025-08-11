# scripts/plot_eval.py
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

RAW_PATH = "data/raw/processed.cleveland.data"
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
MODEL_PATH = "artifacts/model.joblib"
METRICS_JSON = "artifacts/metrics.json"
PLOTS_DIR = Path("artifacts/plots")


def load_cfg(path="configs/train.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data():
    df = pd.read_csv(RAW_PATH, header=None, names=COLS, na_values=["?"])
    df["target"] = (df["num"].astype(float) > 0).astype(int)
    X = df.drop(columns=["num", "target"])
    y = df["target"].values
    return X, y


def get_splits(cfg, X, y):
    test_val = cfg["split"]["test_size"] + cfg["split"]["val_size"]
    stratify_flag = cfg["split"].get("stratify", True)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X,
        y,
        test_size=test_val,
        random_state=cfg["seed"],
        stratify=y if stratify_flag else None,
    )
    rel = cfg["split"]["val_size"] / test_val
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp,
        y_tmp,
        test_size=1 - rel,
        random_state=cfg["seed"],
        stratify=y_tmp if stratify_flag else None,
    )
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def pr_auc_score(y_true, prob):
    P, R, _ = precision_recall_curve(y_true, prob)
    return auc(R, P)


def plot_roc(y_true, prob, title, outpath):
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return float(roc_auc)


def plot_pr(y_true, prob, title, outpath):
    P, R, _ = precision_recall_curve(y_true, prob)
    pr_auc = auc(R, P)
    plt.figure()
    plt.plot(R, P, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return float(pr_auc)


def cm_metrics(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-9)
    return {
        "threshold_used": float(thr),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "specificity": float(specificity),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }, cm


def plot_cm(cm, title, outpath):
    disp = ConfusionMatrixDisplay(cm)
    plt.figure()
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg()
    X, y = load_data()
    _, _, X_val, y_val, X_te, y_te = get_splits(cfg, X, y)

    model = joblib.load(MODEL_PATH)
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        info = json.load(f)

    # รองรับทั้งโครงสร้างเก่า/ใหม่ของ metrics.json
    thr = info.get("threshold")
    if thr is None and isinstance(info.get("thresholds"), dict):
        thr = info["thresholds"].get("f1_optimal") or info["thresholds"].get(
            "recall_constrained", 0.5
        )
    thr = 0.5 if thr is None else float(thr)

    # Probabilities
    val_prob = model.predict_proba(X_val)[:, 1]
    te_prob = model.predict_proba(X_te)[:, 1]

    # ---- ROC / PR (val & test) ----
    roc_val = plot_roc(
        y_val, val_prob, "ROC Curve (Val)", PLOTS_DIR / "roc_curve_val.png"
    )
    roc_te = plot_roc(
        y_te, te_prob, "ROC Curve (Test)", PLOTS_DIR / "roc_curve_test.png"
    )
    pr_val = plot_pr(
        y_val, val_prob, "Precision–Recall Curve (Val)", PLOTS_DIR / "pr_curve_val.png"
    )
    pr_te = plot_pr(
        y_te, te_prob, "Precision–Recall Curve (Test)", PLOTS_DIR / "pr_curve_test.png"
    )

    # ---- Metrics & Confusion Matrix at chosen threshold ----
    val_detail, cm_val = cm_metrics(y_val, val_prob, thr)
    test_detail, cm_test = cm_metrics(y_te, te_prob, thr)

    plot_cm(
        cm_val,
        f"Confusion Matrix (Val) @ thr={thr:.3f}",
        PLOTS_DIR / "confusion_matrix_val.png",
    )
    plot_cm(
        cm_test,
        f"Confusion Matrix (Test) @ thr={thr:.3f}",
        PLOTS_DIR / "confusion_matrix_test.png",
    )

    # รวม metrics เพิ่ม ROC/PR AUC
    val_detail.update(
        {
            "roc_auc": float(roc_auc_score(y_val, val_prob)),
            "pr_auc": float(pr_auc_score(y_val, val_prob)),
        }
    )
    test_detail.update(
        {
            "roc_auc": float(roc_auc_score(y_te, te_prob)),
            "pr_auc": float(pr_auc_score(y_te, te_prob)),
        }
    )

    out = {
        "threshold_used": thr,
        "val": val_detail,
        "test": test_detail,
        "summary": {
            "roc_auc": {"val": roc_val, "test": roc_te},
            "pr_auc": {"val": pr_val, "test": pr_te},
        },
        "plots": {
            "roc_val": str((PLOTS_DIR / "roc_curve_val.png").as_posix()),
            "roc_test": str((PLOTS_DIR / "roc_curve_test.png").as_posix()),
            "pr_val": str((PLOTS_DIR / "pr_curve_val.png").as_posix()),
            "pr_test": str((PLOTS_DIR / "pr_curve_test.png").as_posix()),
            "cm_val": str((PLOTS_DIR / "confusion_matrix_val.png").as_posix()),
            "cm_test": str((PLOTS_DIR / "confusion_matrix_test.png").as_posix()),
        },
    }

    with open(PLOTS_DIR / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved plots to:", str(PLOTS_DIR.resolve()))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
