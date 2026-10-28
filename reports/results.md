# Results

Rows: 151,445 raw, 148,470 after removing the top and bottom 1% per year.

## Median salary by experience (USD)

| Level | Median |
|---|---|
| Entry | $90,000 |
| Mid | $130,000 |
| Senior | $160,000 |
| Executive | $187,000 |

## Median salary by job family (USD)

| Family | Median | Responses |
|---|---|---|
| Machine Learning / AI | $175,275 | 13,574 |
| Research | $170,112 | 8,934 |
| Software Engineering | $165,033 | 35,875 |
| Management | $161,270 | 15,002 |
| Data Science | $150,000 | 18,564 |
| Data Engineering | $143,100 | 20,620 |
| Other | $126,100 | 10,516 |
| Analytics / BI | $104,500 | 25,385 |

## Median salary by work mode (USD)

| Mode | Median |
|---|---|
| Hybrid | $77,098 |
| On-site | $147,000 |
| Remote | $145,000 |

## Model

Gradient boosting on log salary, 118,776 training rows, 29,694 test rows.

| | Mean absolute error |
|---|---|
| Baseline (median by experience and family) | $45,760 |
| Gradient boosting | $43,362 |

R^2 on log salary: 0.343
