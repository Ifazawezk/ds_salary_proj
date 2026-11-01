"""Loading and cleaning the aijobs.net salary survey."""

import re
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "salaries.csv"

EXPERIENCE = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
EMPLOYMENT = {"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}
COMPANY_SIZE = {"S": "Small", "M": "Medium", "L": "Large"}
REMOTE = {0: "On-site", 50: "Hybrid", 100: "Remote"}

# Order matters: the first pattern that matches wins.
TITLE_FAMILIES = [
    ("Machine Learning / AI", r"machine learning|\bml\b|\bai\b|deep learning|computer vision|\bnlp\b|\bllm"),
    ("Data Engineering", r"data engineer|etl|pipeline|big data|data architect|data platform|analytics engineer"),
    ("Data Science", r"data scien"),
    ("Research", r"research|scientist"),
    ("Analytics / BI", r"analyst|analytics|\bbi\b|business intelligence|insight|reporting"),
    ("Management", r"manager|\bhead\b|director|\blead\b|chief|\bvp\b|principal"),
    ("Software Engineering", r"software|developer|engineer"),
]


def title_family(title: str) -> str:
    """Group one of ~400 raw job titles into a broad job family."""
    t = title.lower()
    for family, pattern in TITLE_FAMILIES:
        if re.search(pattern, t):
            return family
    return "Other"


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Decode the survey codes, add job families and drop extreme outliers.

    Rows identical in every column are counted once. About half of the file is
    exact copies, up to 195 of a single row, and they would otherwise weigh on
    every median and put the same row on both sides of a train/test split.

    Salaries outside the 1st-99th percentile of each year are dropped, since a
    few self-reported values (e.g. a wrong currency) distort the averages.
    """
    out = df.drop_duplicates().copy()
    out["experience"] = out["experience_level"].map(EXPERIENCE)
    out["employment"] = out["employment_type"].map(EMPLOYMENT)
    out["company_size_label"] = out["company_size"].map(COMPANY_SIZE)
    out["work_mode"] = out["remote_ratio"].map(REMOTE).fillna("Hybrid")
    out["family"] = out["job_title"].map(title_family)
    out["same_country"] = out["employee_residence"] == out["company_location"]

    bounds = out.groupby("work_year")["salary_in_usd"].quantile([0.01, 0.99]).unstack()
    lo = out["work_year"].map(bounds[0.01])
    hi = out["work_year"].map(bounds[0.99])
    out = out[(out["salary_in_usd"] >= lo) & (out["salary_in_usd"] <= hi)]
    return out.reset_index(drop=True)


def load(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))
