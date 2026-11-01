"""Run the full analysis: clean the data, draw the figures, train the model, write the report."""

from pathlib import Path

from ds_salary import plots
from ds_salary.data import load, load_raw
from ds_salary.model import train_and_evaluate

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
REPORT = ROOT / "reports" / "results.md"


def main():
    raw = load_raw()
    df = load()
    FIG.mkdir(exist_ok=True)
    REPORT.parent.mkdir(exist_ok=True)

    plots.by_experience(df, FIG / "salary_by_experience.png")
    plots.by_family(df, FIG / "salary_by_family.png")
    plots.trend(df, FIG / "salary_trend.png")
    plots.remote(df, FIG / "salary_remote.png")
    plots.countries(df, FIG / "salary_by_country.png")

    _, ev = train_and_evaluate(df)

    exp = df.groupby("experience")["salary_in_usd"].median()
    fam = df.groupby("family")["salary_in_usd"].median().sort_values(ascending=False)
    mode = df.groupby("work_mode")["salary_in_usd"].median()
    lines = [
        "# Results",
        "",
        f"Rows: {len(raw):,} in the file, {len(raw.drop_duplicates()):,} distinct, "
        f"{len(df):,} after removing the top and bottom 1% per year.",
        "",
        "## Median salary by experience (USD)",
        "",
        "| Level | Median |",
        "|---|---|",
        *[f"| {k} | ${exp[k]:,.0f} |" for k in plots.EXP_ORDER],
        "",
        "## Median salary by job family (USD)",
        "",
        "| Family | Median | Responses |",
        "|---|---|---|",
        *[f"| {k} | ${v:,.0f} | {(df['family'] == k).sum():,} |" for k, v in fam.items()],
        "",
        "## Median salary by work mode (USD)",
        "",
        "| Mode | Median |",
        "|---|---|",
        *[f"| {k} | ${v:,.0f} |" for k, v in mode.items()],
        "",
        "## Model",
        "",
        f"Gradient boosting on log salary, trained on {ev.n_train:,} rows from the years before "
        f"{ev.test_year}. Half of {ev.test_year} ({ev.n_calibration:,} rows) sets the salary ranges "
        f"and the other half ({ev.n_test:,} rows) gives the figures below.",
        "",
        "| | Mean absolute error |",
        "|---|---|",
        f"| Baseline (median by experience and family) | ${ev.mae_baseline:,.0f} |",
        f"| Baseline (median by experience, family and country) | ${ev.mae_fair_baseline:,.0f} |",
        f"| Gradient boosting | ${ev.mae_model:,.0f} |",
        "",
        f"Gain over the country baseline: ${ev.gain_low:,.0f} to ${ev.gain_high:,.0f} "
        "(95% bootstrap interval).",
        "",
        f"R^2 on log salary: {ev.r2_log:.3f}",
        "",
        "## Error and range by market",
        "",
        f"The range is meant to hold 8 in 10 salaries. Over all test rows it held {ev.coverage:.1%}.",
        "",
        "| Market | Test rows | Median salary | Mean absolute error | Half of estimates within | Salaries inside the range |",
        "|---|---|---|---|---|---|",
        *[
            f"| {m.market} | {m.rows:,} | ${m.median_salary:,.0f} | ${m.mae:,.0f} | {m.typical_error:.0%} | {m.coverage:.0%} |"
            for m in ev.markets
        ],
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
