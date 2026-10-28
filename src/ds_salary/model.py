"""Salary model: gradient boosting on log salary, compared with a median baseline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAL = [
    "experience",
    "employment",
    "family",
    "company_location",
    "company_size_label",
    "work_mode",
    "same_country",
]
FEATURES = CATEGORICAL + ["work_year"]


@dataclass
class Evaluation:
    mae_model: float
    mae_baseline: float
    r2_log: float
    n_train: int
    n_test: int


def build_pipeline() -> Pipeline:
    encode = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(
        categorical_features=list(range(len(CATEGORICAL))),
        max_iter=300,
        learning_rate=0.08,
        random_state=42,
    )
    return Pipeline([("encode", encode), ("model", model)])


def _features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].astype({"same_country": str})


def train_and_evaluate(df: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, Evaluation]:
    X = _features(df)
    y = np.log(df["salary_in_usd"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    pipe = build_pipeline().fit(X_train, y_train)
    pred = np.exp(pipe.predict(X_test))
    actual = np.exp(y_test)

    # Baseline: the median salary for the same experience level and job family.
    train = X_train.assign(salary=np.exp(y_train))
    medians = train.groupby(["experience", "family"])["salary"].median()
    overall = train["salary"].median()
    baseline = np.array([medians.get(k, overall) for k in zip(X_test["experience"], X_test["family"])])

    ev = Evaluation(
        mae_model=mean_absolute_error(actual, pred),
        mae_baseline=mean_absolute_error(actual, baseline),
        r2_log=r2_score(y_test, np.log(pred)),
        n_train=len(X_train),
        n_test=len(X_test),
    )
    return pipe, ev


def predict(pipe: Pipeline, **profile) -> float:
    """Estimate a salary in USD for one profile (keys as in FEATURES)."""
    return float(np.exp(pipe.predict(_features(pd.DataFrame([profile]))))[0])
