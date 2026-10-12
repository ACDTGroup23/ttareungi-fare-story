# Fare Hike, Same Ride

**Did Seoul's 2023 and 2025 public-transport fare rises push commuters onto 따릉이 (Ddareungi) public bikes?**
Group 23 · ACDT 2026 Semester 2 · Project 1 "Revisit Tackling the Conventional Wisdom" · Hanyang University

> Conventional wisdom: *"따릉이 is mainly for leisure and exercise, not a real commuting mode, so subway fare changes will not meaningfully change how it is used."*

**Answer:** the conclusion holds, the premise does not. Two fare rises (+18.4% in 2023, +5.2% in 2025) moved the weekday commuting-hour share of rides by +0.71 and −0.79 percentage points, both inside our pre-set ±3-point band (equivalence test p = 0.0003 and 0.0017), and every causal design agrees. But 따릉이 is not mainly a leisure mode: rest days have about 25,700 fewer rides than workdays at the same weather. It behaves like a fair-weather option, not a subway substitute.

- **Data story website:** the `docs/` folder, published with GitHub Pages (link in the repository's About box)
- **Executive summary, Three System Maps report, pitch script:** `docs/downloads/`

## Repository layout

| Path | What it is |
|---|---|
| `docs/` | The data story website (GitHub Pages). `index.html` is self-contained; Plotly is vendored in `docs/assets/`. |
| `docs/downloads/` | Executive summary (PDF), Three System Maps report (PDF), pitch script (PDF), notebook, data-and-code bundle, result tables |
| `analysis/Group23_analysis.ipynb` | The full analysis, already executed: 13 figures, every test |
| `analysis/w5_pipeline.py` | The same code as a Python module (the notebook is built from it) |
| `analysis/Group23_condense_ttareungi.py` | Turns the 12 monthly hourly files (200–450 MB each) into the weekday × station table in `data/` |
| `data/` | Raw inputs used by the notebook (see below) |
| `results/` | Every result table as CSV, and the figures as PNG |

## Run it

1. Open `analysis/Group23_analysis.ipynb` in Google Colab, run the first cell and upload `docs/downloads/Group23_data_and_code.zip` when asked. Then *Runtime → Run all* (about 2 minutes).
2. Or locally: `pip install pandas numpy scipy statsmodels scikit-learn matplotlib openpyxl`, put the notebook next to the `data/` folder, and run all cells.

## Data

| File | Source | Coverage |
|---|---|---|
| `data/kma_seoul_108_daily.csv` | Korea Meteorological Administration, ASOS daily, Seoul station 108 | 2022-01-01 to 2025-09-30 |
| `data/daily/*.csv` | Seoul Open Data Plaza OA-14994, public bike daily rentals | 2022 |
| `data/station_day_cache.csv.gz` | Seoul Open Data Plaza OA-15245, public bike hourly usage, condensed to weekday × station (680,258 rows) | Oct–Dec 2022 and 2023, Jul–Sep 2024 and 2025 |
| `data/stations/station_master.xlsx` | Seoul Open Data Plaza OA-13252, public bike station list | 2,789 stations |

## Methods in one paragraph

Outcome: the share of each weekday's rides that start between 07:00–08:59 or 18:00–19:59. Comparison: the same calendar dates one year apart around each fare rise; one weekday is one observation; holidays and subway-strike days are excluded. Primary test: two-sided Welch t-test on temperature-adjusted daily shares, with a TOST equivalence test at ±3 points. Supporting tests: multiple regression with temperature, rain and weekday (HAC errors), Mann–Whitney, χ² on days and on rides, variance F and Levene, power analysis. Causal checks: difference-in-differences against park stations and all other stations, station fixed effects, a 2,000-draw permutation placebo, parallel-trend and Chow tests, dose–response. Daily-volume model: multiple regression with temperature (cubic), log rain and rest day (R² 0.86), compared step by step with the simple regression. Map C: random forests with permutation importance.

## Team

AN Jaewon · LEE Bomin · YUN Sangcheol · YANG Shinje · YOON Sobin · CHOI Brandon Hyungu
