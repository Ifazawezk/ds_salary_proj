"""Salary model: gradient boosting on log salary, judged on the latest year of data.

The model never sees the latest year. Half of that year sets how wide the salary range is
for each country (calibration) and the other half is used only to report the results (test).
So every figure in the report comes from salaries the model and the ranges have not seen, and
the estimator that gets evaluated is the same one the command-line tool uses.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
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

# The range is meant to hold 8 in 10 salaries.
RANGE_QUANTILES = (0.1, 0.9)
# A country needs this many calibration rows to get its own range. Others share one.
MIN_COUNTRY_ROWS = 100
OTHER_COUNTRIES = "Other countries"


@dataclass
class Spread:
    """How far real salaries fell from the estimate in one market, on rows the model did not see."""

    low: float  # multiply the estimate by this for the bottom of the range
    high: float  # multiply the estimate by this for the top of the range
    typical_error: float  # half of the estimates were within this fraction of the real salary


@dataclass
class MarketResult:
    market: str
    rows: int
    median_salary: float
    mae: float
    typical_error: float
    coverage: float


@dataclass
class Evaluation:
    mae_model: float
    mae_baseline: float  # median by experience and job family
    mae_fair_baseline: float  # median by experience, job family and country
    gain_low: float  # 95% interval for how much better the model is than the fair baseline
    gain_high: float
    r2_log: float
    coverage: float
    n_train: int
    n_calibration: int
    n_test: int
    test_year: int
    markets: list[MarketResult] = field(default_factory=list)


@dataclass
class Estimator:
    pipe: Pipeline
    spreads: dict[str, Spread]

    def market(self, country: str) -> str:
        return country if country in self.spreads else OTHER_COUNTRIES

    def spread(self, country: str) -> Spread:
        return self.spreads[self.market(country)]


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


def split_by_latest_year(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on the earlier years, and share the latest year between calibration and test."""
    latest = df["work_year"].max()
    train = df[df["work_year"] < latest]
    if train.empty:
        raise ValueError("Need more than one year of data to hold out the latest year")
    held_out = df[df["work_year"] == latest].sample(frac=1, random_state=seed)
    middle = len(held_out) // 2
    return train, held_out.iloc[:middle], held_out.iloc[middle:]


def _spread(actual: np.ndarray, estimate: np.ndarray) -> Spread:
    low, high = np.quantile(actual / estimate, RANGE_QUANTILES)
    return Spread(low=float(low), high=float(high), typical_error=float(np.median(np.abs(actual - estimate) / actual)))


def calibrate(pipe: Pipeline, calibration: pd.DataFrame) -> dict[str, Spread]:
    """Measure each market's spread on rows the model has not seen."""
    estimate = np.exp(pipe.predict(_features(calibration)))
    actual = calibration["salary_in_usd"].to_numpy()
    country = calibration["company_location"].to_numpy()

    counts = calibration["company_location"].value_counts()
    own_range = set(counts[counts >= MIN_COUNTRY_ROWS].index)
    spreads = {c: _spread(actual[country == c], estimate[country == c]) for c in own_range}

    rest = ~np.isin(country, list(own_range))
    # With no small markets to pool, the shared range falls back on every row.
    spreads[OTHER_COUNTRIES] = _spread(actual[rest], estimate[rest]) if rest.any() else _spread(actual, estimate)
    return spreads


def _median_lookup(train: pd.DataFrame, test: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Median salary for matching rows, falling back on the country and then on everything."""
    medians = train.groupby(keys)["salary_in_usd"].median()
    by_country = train.groupby("company_location")["salary_in_usd"].median()
    overall = train["salary_in_usd"].median()
    return np.array(
        [
            medians.get(key, by_country.get(country, overall))
            for key, country in zip(zip(*(test[k] for k in keys)), test["company_location"])
        ]
    )


def _gain_interval(actual: np.ndarray, model: np.ndarray, baseline: np.ndarray, seed: int) -> tuple[float, float]:
    """95% bootstrap interval for the drop in mean absolute error from using the model."""
    gain = np.abs(actual - baseline) - np.abs(actual - model)
    rng = np.random.default_rng(seed)
    means = [rng.choice(gain, size=len(gain)).mean() for _ in range(1000)]
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def what_drives_pay(pipe: Pipeline, test: pd.DataFrame, seed: int = 42, rows: int = 6000) -> dict[str, float]:
    """How much accuracy (R^2 on log salary) is lost when each feature is shuffled, largest first."""
    sample = test.sample(min(rows, len(test)), random_state=seed)
    result = permutation_importance(
        pipe, _features(sample), np.log(sample["salary_in_usd"]), n_repeats=3, random_state=seed
    )
    importance = dict(zip(FEATURES, result.importances_mean.tolist()))
    return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))


def train_and_evaluate(df: pd.DataFrame, seed: int = 42) -> tuple[Estimator, Evaluation]:
    train, calibration, test = split_by_latest_year(df, seed)

    pipe = build_pipeline().fit(_features(train), np.log(train["salary_in_usd"]))
    estimator = Estimator(pipe=pipe, spreads=calibrate(pipe, calibration))

    pred = np.exp(pipe.predict(_features(test)))
    actual = test["salary_in_usd"].to_numpy()
    baseline = _median_lookup(train, test, ["experience", "family"])
    fair_baseline = _median_lookup(train, test, ["experience", "family", "company_location"])

    markets = np.array([estimator.market(c) for c in test["company_location"]])
    low = pred * np.array([estimator.spreads[m].low for m in markets])
    high = pred * np.array([estimator.spreads[m].high for m in markets])
    inside = (actual >= low) & (actual <= high)
    gain_low, gain_high = _gain_interval(actual, pred, fair_baseline, seed)

    results = [
        MarketResult(
            market=str(m),
            rows=int((markets == m).sum()),
            median_salary=float(np.median(actual[markets == m])),
            mae=mean_absolute_error(actual[markets == m], pred[markets == m]),
            typical_error=_spread(actual[markets == m], pred[markets == m]).typical_error,
            coverage=float(inside[markets == m].mean()),
        )
        for m in np.unique(markets)
    ]
    results.sort(key=lambda r: r.rows, reverse=True)

    ev = Evaluation(
        mae_model=mean_absolute_error(actual, pred),
        mae_baseline=mean_absolute_error(actual, baseline),
        mae_fair_baseline=mean_absolute_error(actual, fair_baseline),
        gain_low=gain_low,
        gain_high=gain_high,
        r2_log=r2_score(np.log(actual), np.log(pred)),
        coverage=float(inside.mean()),
        n_train=len(train),
        n_calibration=len(calibration),
        n_test=len(test),
        test_year=int(test["work_year"].iloc[0]),
        markets=results,
    )
    return estimator, ev


def predict(pipe: Pipeline, **profile) -> float:
    """Estimate a salary in USD for one profile (keys as in FEATURES)."""
    return float(np.exp(pipe.predict(_features(pd.DataFrame([profile]))))[0])
