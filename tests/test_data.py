import pandas as pd
import pytest

from ds_salary.data import clean, title_family
from ds_salary.model import predict, train_and_evaluate


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


def test_model_learns_experience():
    pipe, ev = train_and_evaluate(clean(_sample(2000)))
    base = dict(work_year=2025, employment="Full-time", family="Data Science", company_location="US",
                company_size_label="Medium", work_mode="Remote", same_country=True)
    assert predict(pipe, experience="Senior", **base) > predict(pipe, experience="Entry", **base)
    # Salary here depends only on experience, so both should be close to exact.
    assert ev.mae_model < 5_000
    assert ev.mae_baseline < 5_000
