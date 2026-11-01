# Results

Rows: 151,445 in the file, 71,913 distinct, 70,492 after removing the top and bottom 1% per year.

## Median salary by experience (USD)

| Level | Median |
|---|---|
| Entry | $88,084 |
| Mid | $122,000 |
| Senior | $155,413 |
| Executive | $188,640 |

## Median salary by job family (USD)

| Family | Median | Responses |
|---|---|---|
| Machine Learning / AI | $168,700 | 6,386 |
| Management | $161,344 | 8,135 |
| Research | $160,000 | 3,850 |
| Software Engineering | $150,000 | 15,083 |
| Data Science | $142,100 | 7,103 |
| Data Engineering | $139,152 | 9,659 |
| Other | $124,824 | 6,606 |
| Analytics / BI | $102,960 | 13,670 |

## Median salary by work mode (USD)

| Mode | Median |
|---|---|
| Hybrid | $75,683 |
| On-site | $138,000 |
| Remote | $140,600 |

## Median salary by work mode, inside and outside the US (USD)

| Mode | US median | US rows | Elsewhere median | Elsewhere rows |
|---|---|---|---|---|
| Hybrid | $115,000 | 52 | $66,794 | 242 |
| On-site | $147,000 | 44,578 | $90,879 | 8,450 |
| Remote | $145,360 | 14,729 | $109,285 | 2,441 |

## Model

Gradient boosting on log salary, trained on 32,972 rows from the years before 2025. Half of 2025 (18,760 rows) sets the salary ranges and the other half (18,760 rows) gives the figures below.

| | Mean absolute error |
|---|---|
| Baseline (median by experience and family) | $49,057 |
| Baseline (median by experience, family and country) | $45,989 |
| Gradient boosting | $45,713 |

Gain over the country baseline: $172 to $387 (95% bootstrap interval).

R^2 on log salary: 0.329

## Error and range by market

The range is meant to hold 8 in 10 salaries. Over all test rows it held 79.8%.

| Market | Test rows | Median salary | Mean absolute error | Half of estimates within | Salaries inside the range |
|---|---|---|---|---|---|
| US | 15,689 | $147,000 | $48,338 | 27% | 80% |
| CA | 1,343 | $119,700 | $35,535 | 24% | 81% |
| GB | 701 | $77,500 | $32,070 | 27% | 80% |
| Other countries | 655 | $57,894 | $28,324 | 36% | 78% |
| AU | 153 | $118,267 | $27,346 | 12% | 76% |
| NL | 136 | $71,842 | $23,461 | 27% | 80% |
| DE | 83 | $91,789 | $37,061 | 34% | 82% |

## What drives pay

How much accuracy (R^2 on log salary) is lost when one feature is shuffled in the test rows.

| Feature | Loss of R^2 |
|---|---|
| company_location | 0.247 |
| experience | 0.184 |
| family | 0.134 |
| employment | 0.005 |
| work_mode | 0.003 |
| company_size_label | 0.000 |
| same_country | 0.000 |
| work_year | 0.000 |
