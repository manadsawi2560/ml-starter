import sys
import json
import joblib
import pandas as pd


def predict(model_path, input_path_or_json):
    model = joblib.load(model_path)
    if input_path_or_json.endswith(".csv"):
        X = pd.read_csv(input_path_or_json)
    else:
        X = pd.DataFrame([json.loads(input_path_or_json)])
    prob = model.predict_proba(X)[:, 1]
    return [{"prob": float(p), "label": int(p >= 0.5)} for p in prob]


if __name__ == "__main__":
    # ใช้: python -m ml_starter.models.predict artifacts/model.joblib sample.csv
    out = predict(sys.argv[1], sys.argv[2])
    print(json.dumps(out, indent=2))
