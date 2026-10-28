"""Figures for the report."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", palette="deep")
EXP_ORDER = ["Entry", "Mid", "Senior", "Executive"]


def _k(v, _):
    return f"${v / 1000:.0f}k"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def by_experience(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="experience", y="salary_in_usd", order=EXP_ORDER, showfliers=False, ax=ax)
    ax.set(title="Salary by experience level", xlabel="", ylabel="Salary (USD)")
    ax.yaxis.set_major_formatter(_k)
    _save(fig, path)


def by_family(df: pd.DataFrame, path):
    med = df.groupby("family")["salary_in_usd"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    med.plot.barh(ax=ax, color=sns.color_palette()[0])
    ax.set(title="Median salary by job family", xlabel="Median salary (USD)", ylabel="")
    ax.xaxis.set_major_formatter(_k)
    _save(fig, path)


def trend(df: pd.DataFrame, path):
    recent = df[df["work_year"] >= 2022]
    med = recent.groupby(["work_year", "experience"])["salary_in_usd"].median().unstack()[EXP_ORDER]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    med.plot(marker="o", ax=ax)
    ax.set(title="Median salary over time", xlabel="Year", ylabel="Median salary (USD)")
    ax.set_xticks(med.index)
    ax.yaxis.set_major_formatter(_k)
    _save(fig, path)


def remote(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x="experience", y="salary_in_usd", hue="work_mode", order=EXP_ORDER,
                estimator="median", errorbar=None, ax=ax)
    ax.set(title="Median salary: remote, hybrid and on-site", xlabel="", ylabel="Median salary (USD)")
    ax.yaxis.set_major_formatter(_k)
    _save(fig, path)


def countries(df: pd.DataFrame, path, top: int = 10):
    counts = df["company_location"].value_counts()
    keep = counts[counts >= 100].index
    med = df[df["company_location"].isin(keep)].groupby("company_location")["salary_in_usd"].median()
    med = med.sort_values(ascending=False).head(top).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    med.plot.barh(ax=ax, color=sns.color_palette()[2])
    ax.set(title=f"Median salary by company country (top {top}, 100+ responses)",
           xlabel="Median salary (USD)", ylabel="")
    ax.xaxis.set_major_formatter(_k)
    _save(fig, path)
