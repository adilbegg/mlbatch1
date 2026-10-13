# Project 3 — Statistical Storytelling

## Does Clear Weather Justify Higher Bike-Sharing Capacity?

This project examines whether daily bicycle rental demand differs between **clear** and **non-clear** weather conditions by enough to justify *considering* a 10% increase in planned bicycle availability. It compares the unadjusted difference with estimates adjusted for temperature and year, quantifies uncertainty, investigates alternative explanations, and tests sensitivity to analytical choices.

**Decision question:** Does average daily bike-sharing demand differ between clear and non-clear weather days by enough to justify considering at least 10% more bicycle availability on clear-weather days, after accounting for temperature?

> This is an observational analysis of recorded rentals, not an estimate of the causal effect of weather or a calculation of the optimal bicycle fleet size. The 10% threshold is an illustrative decision rule, not a demonstrated operating optimum.

## Dataset and licence

- **Dataset:** [Bike Sharing Dataset, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- **File analysed:** `day.csv` (daily observations)
- **Source system:** Capital Bikeshare, Washington, D.C.
- **Coverage:** 1 January 2011 to 31 December 2012 (731 days)
- **Licence:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Reuse requires appropriate attribution.
- **Dataset citation:** Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894

Each row represents **one calendar day**. The numeric outcome is `cnt`, total recorded bicycle rentals (casual plus registered users). The grouping variable is derived from `weathersit`: category 1 (clear/few clouds/partly cloudy) is labelled `Clear`; categories 2–4 are labelled `Non-Clear`. The dataset's `temp` variable is **normalised**, not recorded directly in degrees Celsius. `yr=0` represents 2011 and `yr=1` represents 2012.

## Project files

```text
week03/
├── data/
│   └── day.csv
├── project3_analysis.ipynb
├── REPORT.md
└── README.md
```

The `data/day.csv` file should be placed in this location before running the notebook. Obtain it from the UCI dataset download if it is not included in the repository.

## How to run in VS Code

1. Install Python 3, VS Code, and Microsoft's **Python** and **Jupyter** extensions.
2. Open the `week03` directory in VS Code.
3. Place the UCI `day.csv` file under `data/day.csv`.
4. Create and activate an optional virtual environment:

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows PowerShell**

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

5. Install required packages:

   ```bash
   python -m pip install pandas numpy matplotlib scipy jupyter ipykernel
   ```

6. Open `project3_analysis.ipynb`, choose the corresponding Python kernel, then **Restart Kernel** and **Run All**.

The notebook uses seed **42** for its bootstrap random-number generator. All figures and printed summaries should regenerate from the original CSV. If running from a different directory, ensure that the relative path `data/day.csv` is correct.

## Analysis outline

1. Audit completeness, duplicates, date coverage, rental totals, and unusual values.
2. Compare mean daily rentals in the clear and non-clear groups; calculate a Welch 95% confidence interval.
3. Assess temperature as a plausible confounder by comparing group composition and rental associations.
4. Calculate an adjusted estimate by comparing weather groups within four temperature quantile bands and weighting band-specific differences by the overall distribution of days.
5. Estimate the adjusted interval using 2,000 stratified bootstrap resamples.
6. Check sensitivity using three bands, five bands, and fixed-width bands.
7. Add **year** as a second stratification variable and bootstrap the temperature-plus-year estimate.
8. Interpret the statistical evidence against an illustrative threshold of 10% of the observed non-clear-day mean.

## Headline results

| Comparison | Estimated difference (additional rentals/day) | 95% confidence interval |
|---|---:|---:|
| Naive, clear minus non-clear | +1,015.86 | 733.54 to 1,298.19 (Welch) |
| Adjusted for temperature bands | +851.92 | 628.66 to 1,072.28 (bootstrap) |
| Adjusted for temperature bands and year | **+779.63** | **630.36 to 932.61 (bootstrap)** |

The main results are associations rather than causal effects. The confidence intervals rely on simplifying assumptions about observations and resampling and do not explicitly model serial dependence in the daily time series.

See [REPORT.md](REPORT.md) for the decision-oriented discussion, sensitivity estimates, alternative explanations, limitations, and evidence that could change the conclusion.
