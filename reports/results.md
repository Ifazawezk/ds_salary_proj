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

## Model

Gradient boosting on log salary, 56,393 training rows, 14,099 test rows.

| | Mean absolute error |
|---|---|
| Baseline (median by experience and family) | $48,751 |
| Gradient boosting | $45,157 |

R^2 on log salary: 0.359
