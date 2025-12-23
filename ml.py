import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

FEATURE_COLS = ["tenure", "monthly_charges", "contract"]

def train_model(data_path: Path, model_path: Path) -> Pipeline:
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS]
    y = df["churn"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["tenure", "monthly_charges"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["contract"]),
        ]
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", LogisticRegression(max_iter=200))])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42)
    pipe.fit(X_train, y_train)

    joblib.dump(pipe, model_path)
    return pipe

def load_or_train(data_path: Path, model_path: Path) -> Pipeline:
    if model_path.exists():
        return joblib.load(model_path)
    return train_model(data_path, model_path)
