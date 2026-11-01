# What do AI, ML and data jobs pay?

An analysis of **about 72,000 distinct salary rows** from the [aijobs.net salary survey](https://aijobs.net/salaries/) (2020 to 2025), plus a small model and command-line tool that estimates a salary for a given role, level and country.

![Median salary by job family](figures/salary_by_family.png)

## Key findings

| | |
|---|---|
| **Experience matters most** | Median pay roughly doubles from Entry ($88k) to Executive ($189k). |
| **ML / AI roles pay the most** | ML and AI roles have a median of $169k, against $103k for analytics and BI roles. |
| **Remote pays about the same as on-site** | Median $141k remote vs $138k on-site. Hybrid responses are lower ($76k), mostly because they come from smaller markets. |
| **Location is the second biggest factor** | US companies pay far more than the rest of the sample. 84% of rows come from US companies, so treat other countries with care. |

Full numbers: [reports/results.md](reports/results.md).

<p>
  <img src="figures/salary_by_experience.png" width="49%">
  <img src="figures/salary_trend.png" width="49%">
  <img src="figures/salary_remote.png" width="49%">
  <img src="figures/salary_by_country.png" width="49%">
</p>

## Estimate a salary

```bash
pip install -e .
ds-salary --experience Senior --family "Data Science" --country GB --mode Remote
```

```
Estimated salary: $106,000 per year
Likely range: $65,000 to $190,000
(On unseen 2025 data for GB, half of the estimates were within 30%. Rows from companies in GB: 2,738.)
```

Options: `--experience` (Entry, Mid, Senior, Executive), `--family`, `--country` (ISO code), `--size` (Small, Medium, Large), `--mode` (On-site, Hybrid, Remote), `--employment`, `--year`, `--abroad`.

`--year` changes nothing after 2024: the model only learns from earlier years and cannot predict a trend beyond them, so 2024, 2025 and 2030 all give the same answer.

## How it works

1. **Cleaning** ([`data.py`](src/ds_salary/data.py)): counts identical rows once, decodes the survey codes, groups 400+ job titles into 8 job families, and drops the top and bottom 1% of salaries each year to remove obvious entry errors.
2. **Figures** ([`plots.py`](src/ds_salary/plots.py)): the five charts above.
3. **Model** ([`model.py`](src/ds_salary/model.py)): gradient boosting (`HistGradientBoostingRegressor`) on log salary, using experience, job family, country, company size, work mode, employment type and year. It is trained only on the years before 2025. Half of 2025 sets the salary ranges and the other half gives the figures below, so these are salaries the model has not seen.

| | Mean absolute error |
|---|---|
| Baseline (median by experience and family) | $49,057 |
| Baseline (median by experience, family and country) | $45,989 |
| Gradient boosting | $45,713 |

**The model adds very little.** Looking up the median for the same experience, job family and country gets almost the same result: the model is only $172 to $387 better (95% bootstrap interval). An earlier version of this page compared it only with the first baseline, which does not know the country, and that made the gain look like $2,400. Most of the spread in pay comes from things the survey does not record, such as the company, the city and the person's skills. Use the estimate as a rough range, not a number to negotiate on.

**The range is measured, not assumed.** For each country with at least 100 calibration rows, it runs from the 10th to the 90th percentile of real salary divided by the estimate. Smaller countries share one range. It is meant to hold 8 in 10 salaries, and on the test half it held 79.8%.

| Market | Test rows | Median salary | Mean absolute error | Half of estimates within | Salaries inside the range |
|---|---|---|---|---|---|
| US | 15,689 | $147,000 | $48,338 | 27% | 80% |
| CA | 1,343 | $119,700 | $35,535 | 24% | 81% |
| GB | 701 | $77,500 | $32,070 | 27% | 80% |
| Other countries | 655 | $57,894 | $28,324 | 36% | 78% |
| AU | 153 | $118,267 | $27,346 | 12% | 76% |
| NL | 136 | $71,842 | $23,461 | 27% | 80% |
| DE | 83 | $91,789 | $37,061 | 34% | 82% |

Three limits on these figures:

- The top and bottom 1% of salaries are removed before any of this. If those rows count as misses, the range held about 78%.
- 21.6% of the test rows match a row from an earlier year in every column except the year, probably the same posting again. Leaving them out moves the error from $45,713 to $45,975.
- A country outside the table gets the shared range, and its estimate leans on other countries. Qatar has one row in the file, so `--country QA` is little more than a guess, and the tool warns about this.

## Run it yourself

```bash
pip install -e ".[dev]"
python run_analysis.py   # rebuilds figures/ and reports/results.md, about 20 seconds
pytest                   # tests
```

## Data

`data/salaries.csv` is a snapshot of [foorilla/ai-jobs-net-salaries](https://github.com/foorilla/ai-jobs-net-salaries), published under CC0 (public domain). Salaries are self-reported and anonymous, and most come from the US, so results for other countries are less reliable.

The file has 151,445 rows but only 71,913 distinct ones: 52.5% are exact copies, and one row appears 195 times. From 2022 on, more than half of the rows also sit next to a row that matches on everything except salary, and the first of the two is usually about 1.5 times the second. That looks like the top and bottom of a posted pay range rather than two people answering a survey, although the file does not say so. Identical rows are counted once here. Pairs that differ in salary are kept, so read these figures as a mix of reported and posted pay.

## License

Code: MIT. Data: CC0.
