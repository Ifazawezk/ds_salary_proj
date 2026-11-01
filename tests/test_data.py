import re

import numpy as np
import pandas as pd
import pytest

from ds_salary.cli import main
from ds_salary.data import clean, title_family
from ds_salary.model import (
    OTHER_COUNTRIES,
    calibrate,
    predict,
    split_by_latest_year,
    train_and_evaluate,
    what_drives_pay,
)


@pytest.mark.parametrize(
    "title, family",
    [
        ("Data Scientist", "Data Science"),
        ("Machine Learning Engineer", "Machine Learning / AI"),
        ("AI Engineer", "Machine Learning / AI"),
        ("Data Engineer", "Data Engineering"),
        ("Analytics Engineer", "Data Engineering"),
        ("Business Intelligence Analyst", "Analytics / BI"),
        ("Research Scientist", "Research"),
        ("Software Engineer", "Software Engineering"),
        ("Engineering Manager", "Management"),
        ("Associate", "Other"),
    ],
)
def test_title_family(title, family):
    assert title_family(title) == family


def _sample(n=400):
    rows = []
    for i in range(n):
        level = ["EN", "MI", "SE", "EX"][i % 4]
        rows.append({
            "work_year": 2024 + i % 2,
            "experience_level": level,
            "employment_type": "FT",
            "job_title": ["Data Scientist", "Data Analyst"][i % 2],
            "salary": 0,
            "salary_currency": "USD",
            "salary_in_usd": 60_000 + 30_000 * "EN MI SE EX".split().index(level) + i,
            "employee_residence": "US",
            "remote_ratio": [0, 50, 100][i % 3],
            "company_location": "US",
            "company_size": "M",
        })
    return pd.DataFrame(rows)


def test_clean_decodes_and_trims():
    df = clean(_sample())
    assert set(df["experience"]) == {"Entry", "Mid", "Senior", "Executive"}
    assert set(df["work_mode"]) == {"On-site", "Hybrid", "Remote"}
    assert df["same_country"].all()
    assert len(df) < 400  # the 1% tails are removed


def test_clean_counts_identical_rows_once():
    sample = _sample()
    assert len(clean(pd.concat([sample, sample, sample]))) == len(clean(sample))


def test_clean_keeps_rows_that_differ_in_salary():
    sample = _sample()
    other = sample.assign(salary_in_usd=sample["salary_in_usd"] + 1)
    assert len(clean(pd.concat([sample, other]))) > len(clean(sample))


def _years_sample(n=2000):
    """Every title and level in every year, so the latest year looks like the earlier ones."""
    sample = _sample(n)
    sample["work_year"] = [2023 + (i // 2) % 3 for i in range(n)]
    return sample


def _countries_sample(n=3000):
    """Two countries with different pay, and a small one that has to share a range."""
    sample = _years_sample(n)
    rng = np.random.default_rng(0)
    sample["company_location"] = rng.choice(["US", "GB", "QA"], size=n, p=[0.6, 0.38, 0.02])
    sample["employee_residence"] = sample["company_location"]
    pay = {"US": 1.0, "GB": 0.5, "QA": 0.8}
    noise = rng.lognormal(0, 0.2, size=n)
    sample["salary_in_usd"] = (sample["salary_in_usd"] * sample["company_location"].map(pay) * noise).round()
    return sample


def test_model_learns_experience():
    estimator, ev = train_and_evaluate(clean(_years_sample()))
    base = dict(work_year=2025, employment="Full-time", family="Data Science", company_location="US",
                company_size_label="Medium", work_mode="Remote", same_country=True)
    assert predict(estimator.pipe, experience="Senior", **base) > predict(estimator.pipe, experience="Entry", **base)
    # Salary here depends only on experience, so both should be close to exact.
    assert ev.mae_model < 5_000
    assert ev.mae_baseline < 5_000


def test_latest_year_is_never_trained_on_and_is_shared_evenly():
    df = clean(_years_sample())
    train, calibration, test = split_by_latest_year(df)
    assert train["work_year"].max() < calibration["work_year"].min()
    assert set(calibration["work_year"]) == set(test["work_year"]) == {df["work_year"].max()}
    assert abs(len(calibration) - len(test)) <= 1
    assert len(train) + len(calibration) + len(test) == len(df)
    assert not set(calibration.index) & set(test.index)


def test_one_year_of_data_is_refused():
    df = clean(_years_sample())
    with pytest.raises(ValueError, match="more than one year"):
        split_by_latest_year(df[df["work_year"] == 2025])


def test_evaluation_uses_each_row_once():
    df = clean(_years_sample())
    _, ev = train_and_evaluate(df)
    assert ev.n_train + ev.n_calibration + ev.n_test == len(df)
    assert ev.test_year == 2025


def test_range_holds_about_eight_in_ten_salaries():
    _, ev = train_and_evaluate(clean(_countries_sample()))
    assert 0.72 < ev.coverage < 0.88


def test_small_country_shares_a_range_and_large_ones_have_their_own():
    estimator, ev = train_and_evaluate(clean(_countries_sample()))
    assert estimator.market("US") == "US"
    assert estimator.market("GB") == "GB"
    assert estimator.market("QA") == OTHER_COUNTRIES
    assert estimator.market("ZZ") == OTHER_COUNTRIES
    assert {m.market for m in ev.markets} == {"US", "GB", OTHER_COUNTRIES}
    assert sum(m.rows for m in ev.markets) == ev.n_test


def test_range_brackets_the_estimate():
    estimator, _ = train_and_evaluate(clean(_countries_sample()))
    for spread in estimator.spreads.values():
        assert spread.low < 1 < spread.high
        assert 0 < spread.typical_error < 1


def test_shared_range_falls_back_on_every_row_when_no_country_is_small():
    df = clean(_years_sample())
    estimator, _ = train_and_evaluate(df)
    spreads = calibrate(estimator.pipe, df[df["work_year"] == 2025])
    assert set(spreads) == {"US", OTHER_COUNTRIES}
    assert spreads["US"] == spreads[OTHER_COUNTRIES]


def test_fair_baseline_knows_the_country():
    _, ev = train_and_evaluate(clean(_countries_sample()))
    assert ev.mae_fair_baseline < ev.mae_baseline
    assert ev.gain_low <= ev.gain_high


def test_what_drives_pay_finds_the_real_causes():
    df = clean(_countries_sample())
    estimator, _ = train_and_evaluate(df)
    _, _, test = split_by_latest_year(df)
    importance = what_drives_pay(estimator.pipe, test)
    # Pay in this sample depends on country and experience, and on nothing else.
    assert set(list(importance)[:2]) == {"company_location", "experience"}
    assert importance["experience"] > 0.1
    assert importance["work_mode"] < 0.02
    assert list(importance.values()) == sorted(importance.values(), reverse=True)


def test_cli_prints_a_range_around_the_estimate(capsys):
    main(["--experience", "Senior", "--family", "Data Science", "--country", "gb", "--mode", "Remote"])
    out = capsys.readouterr().out
    amounts = [int(a.replace(",", "")) for a in re.findall("[$]([0-9,]+)", out)]
    estimate, low, high = amounts[:3]
    assert low < estimate < high
    assert estimate % 1000 == 0
    assert "for GB" in out


def test_cli_says_when_a_country_shares_a_range(capsys):
    main(["--country", "qa"])
    out = capsys.readouterr().out
    assert "countries with few responses" in out
    assert "Rows from companies in QA: 1." in out
    assert "Warning: few responses" in out
