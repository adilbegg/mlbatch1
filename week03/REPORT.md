# Project 3 — Statistical Storytelling

## Does Clear Weather Justify Higher Bike-Sharing Capacity?

### 1. The Question

Does average daily bicycle rental demand differ between clear and non-clear weather days by enough to justify **considering at least 10% more planned bicycle availability on clear-weather days**, after accounting for temperature? This question matters to a bike-sharing operator deciding how to position bicycles and plan service capacity. The 10% cutoff is an illustrative management threshold, not an empirically established optimal fleet increase. A larger observed rental gap could warrant further planning, but cannot alone establish how many bicycles to add.

### 2. The Answer

**Yes, the data support further consideration of additional availability on clear-weather days, but not an automatic 10% fleet expansion.** In the raw comparison, clear days recorded **1,015.86 more rentals per day** on average (95% Welch confidence interval: **733.54 to 1,298.19**), equivalent to **26.31%** of the non-clear-day mean. After accounting for temperature, the estimated difference was **851.92** (95% bootstrap interval: **628.66 to 1,072.28**). After also controlling for year, it fell to **779.63 additional rentals per day** (95% bootstrap interval: **630.36 to 932.61**).

The observed non-clear-day mean was 3,860.92 rentals; 10% of that baseline is **386.09 rentals/day**. Even the lower bound of the temperature-and-year-adjusted interval exceeds this illustrative benchmark. However, recorded trips do not reveal unmet demand, available bicycles, or the cost of adding capacity. The evidence supports **investigating an operational response**, not declaring a specific fleet increase profitable.

**One number to remember: approximately 780 additional recorded rentals per day on clear-weather days, after accounting for temperature and year.**

### 3. The Data

The analysis uses the publicly available [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), specifically `day.csv`, documenting Capital Bikeshare activity in Washington, D.C. from **1 January 2011 through 31 December 2012**. It is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Citation: Fanaee-T, H. (2013), *Bike Sharing*, UCI Machine Learning Repository, https://doi.org/10.24432/C5W894.

There are **731 rows**, each representing one calendar day, and **16 original columns**. The outcome, `cnt`, is the daily total of casual and registered rentals. Weather categories were combined into **Clear** (category 1, including few clouds and partly cloudy) and **Non-clear** (categories 2–4). The groups contain **463** and **268** days, respectively.

The notebook found **no missing values, duplicate dates, negative rental counts, or discrepancies between `casual + registered` and `cnt`**. Recorded daily rentals range from **22 to 8,714**. Very low values merit caution because they could indicate exceptional conditions; they should not automatically be deleted. The continuous date coverage verifies recorded dates but cannot independently prove that no measurement or system-level problems occurred.

### 4. The Analysis

First, I compared average rentals by weather group without adjustment. Clear days averaged **4,876.79** rentals and non-clear days **3,860.92**. The unadjusted difference was **1,015.86**, or **26.31%** relative to the non-clear baseline. A two-sample Welch interval quantified uncertainty without requiring the two groups to have identical variances.

Second, I examined temperature as a plausible confounder. Temperature is a **normalised** variable in this dataset, not directly degrees Celsius. Its correlation with daily rentals was **0.627**. The clear-weather group had a higher mean normalised temperature (**0.51 versus 0.47**) and was more concentrated in the warmest temperature quartile (**30.5% versus 15.7%**). Thus, directly comparing all clear days with all non-clear days risks mixing a weather-category difference with a difference in temperature composition.

To improve comparability, I divided days into four approximately equal-sized temperature quantile bands. Within each band, I calculated the difference in mean rentals between clear and non-clear days, then combined those differences using the proportion of all days in each band as weights. This direct standardisation produced **851.92 additional rentals/day**, **163.94 lower** than the naive result—a **16.1% reduction** in the original gap. A stratified bootstrap with **2,000 resamples**, resampling within temperature-by-weather groups, provided its uncertainty interval.

Finally, I added year to the stratification, comparing weather groups within combinations of temperature band and **2011 or 2012**. This more stringent comparison produced **779.63 additional rentals/day** and a new 2,000-resample bootstrap interval of **630.36 to 932.61**. These adjusted comparisons describe associations among more similar observed days; they do not identify the causal effect of weather.

### 5. Alternatives Ruled Out

**Chance.** The naive 95% confidence interval (**733.54 to 1,298.19**) and the two-confounder bootstrap interval (**630.36 to 932.61**) both exclude zero. Under their statistical assumptions, the observed positive differences are unlikely to be explained by sampling variation alone. Importantly, excluding zero does not itself show that an operational investment will pay off. The intervals also assume a resampling structure that does not explicitly capture dependence among neighbouring days.

**Confounding.** Temperature differs between the groups and is strongly related to rental demand. The drop from **1,015.86** naive rentals to **851.92** temperature-adjusted rentals shows why the naive comparison cannot be treated as a standalone answer. Year also matters: average daily rentals rose from **3,405.76 in 2011** to **5,599.93 in 2012**, while the year composition differed slightly between weather groups. Controlling for temperature and year together reduced the estimate further to **779.63**. These controls address two plausible explanations, not every possible source of bias.

**Artefact.** The data audit found no recorded missing values, duplicate dates, negative totals, or arithmetic mismatches. The two years are fully represented by unique daily dates. However, these checks cannot establish how consistently weather was classified or whether station outages, equipment shortages, or unmet rental requests affected the recorded counts. There is also a composition issue: a day with fewer recorded rentals may reflect reduced demand, limited availability, or both. Therefore, measurement and availability artefacts **cannot be fully ruled out** from this dataset.

### 6. Sensitivity

I recomputed the temperature-adjusted difference under three alternative definitions of temperature bands:

| Adjustment | Extra rentals/day | Change versus four-band estimate |
|---|---:|---:|
| Four quantile bands (main) | **851.92** | — |
| Three quantile bands | 821.12 | −30.80 |
| Five quantile bands | 783.33 | −68.59 |
| Four fixed-width bands | 757.87 | −94.06 |

The four temperature-only estimates range from **757.87 to 851.92**. All remain positive, but the magnitude drifts downward as the banding choice changes. This argues for **stability in direction, not exact stability in size**. The additional temperature-plus-year adjustment yields **779.63**, close to the lower part of the temperature-only range. Its confidence interval remains positive. Together, these results make the positive association more credible while showing that the naive estimate exaggerates the association remaining after the selected adjustments.

### 7. Limits

The observations are from **one bike-sharing system during 2011–2012**. Travel habits, infrastructure, fleet sizes, and commuting patterns may have changed, so the numeric results should not automatically be applied to another city or to current operations. This is an **observational comparison**: seasonality, holidays, events, user preferences, and other unmeasured influences may still confound the association. Broad temperature bands leave within-band variation; using narrower combinations eventually produces smaller, less reliable subgroups.

A further statistical limitation is that consecutive days may be correlated. The Welch interval and within-stratum bootstrap treat records more independently than a daily time series might warrant, potentially misrepresenting uncertainty. Most importantly, `cnt` measures **completed rentals**, not customer requests turned away, station-level bicycle shortages, or the operating cost of extra capacity. A 10% demand-gap benchmark is not equivalent to proof that fleet capacity should grow by exactly 10%.

### 8. What Would Change It

I would weaken or reverse the recommendation if a time-aware analysis, newer years of data, or finer controls for season, holidays, and system changes reduced the adjusted association substantially or moved its confidence interval across the **386.09-rental benchmark**. I would also reconsider if station-level records showed that clear days already have abundant unused bikes, meaning additional fleet investment would create little benefit. Conversely, verified unmet demand on clear days, combined with a cost-benefit analysis showing that additional repositioning or bicycles would pay for themselves, would justify a firmer operational recommendation. **The next decision should be a targeted capacity and unmet-demand study, not an immediate fleet expansion.**
