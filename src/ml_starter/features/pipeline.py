from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


def build_preprocess():
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, NUMERIC),
            ("cat", cat_pipe, CATEGORICAL),
        ]
    )
