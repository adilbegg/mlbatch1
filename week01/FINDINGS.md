# Project 1 Findings Document

## Dataset

1. This dataset is a real Titanic passenger dataset with 891 rows and 12 columns. It is a sensible EDA dataset because it contains a mix of integer, float, and categorical fields, which means there is room to inspect not just the target but also the structure of the underlying data.

2. The row-level unit is a passenger record. Each row represents one passenger, and PassengerId acts as a unique identifier for the records in this dataset. This matters because a model should never accidentally treat repeated or duplicated passengers as separate observations.

## Data Quality

3. Cabin is the largest missing-data issue in the dataset, with 687 missing values, which is 77.10% of the rows. This is a major data-quality concern because a model relying on cabin information would be forced to ignore most of the records or rely on a brittle imputation strategy. I would keep the missingness visible and treat cabin as a weak or secondary feature unless there is a clear domain reason to engineer a derived grouping.

4. Age is missing for 177 passengers, which is 19.87% of the dataset. This is large enough to affect any analysis using age as a predictor, but it is not so severe that the entire feature becomes unusable. I would document the missingness and handle it only in a later preprocessing step, not during EDA.

5. Embarked is missing in only two rows, or 0.22% of the data. This is a small issue, and it is not likely to distort the overall analysis. I would still note it explicitly and avoid silently deleting those records during exploratory work.

6. There are no duplicate rows and no duplicate PassengerId values. This is helpful because it means the dataset is internally consistent at the record level. It reduces the risk that repeated rows are inflating the signal or causing leakage in later modeling work.

7. There are no negative ages or future dates in the dataset, so the obvious impossible-value checks do not find a serious problem. This is a good sign, but it does not remove the need to monitor missingness and unusual values, especially in fields with strong distributional skew.

8. I did not find the sentinel-style values listed in the brief, such as 999, -1, ?, or N/A, in a way that is obviously meant to stand in for missing data. In this dataset, zero values are common and valid in some fields, so zero on its own should not be treated as a sentinel. I would only treat zero as suspicious if a specific feature rule says it is impossible.

9. The column dtypes are sensible for this dataset. Numeric values are stored as numeric types, and text fields are stored as text. I do not see a clear case where a number is stored as text or a date is stored as text. This keeps the notebook simpler and reduces the risk of silent conversion mistakes during later modeling.

10. There are no truly constant columns, but several variables are near-constant in the sense that they have very few unique values. Survived, Pclass, Sex, SibSp, Parch, and Embarked all show this pattern. This is not a flaw in itself, but it means these variables may be highly imbalanced and should be interpreted carefully, especially in any split or sampling step.

## Target Analysis

11. The target variable is Survived. This is the correct target because it is the variable a model would try to predict and it is directly connected to the business story of the dataset. A model’s value depends on whether survival can be explained by available passenger information.

12. The target is imbalanced but not extreme. The dataset contains more passengers who did not survive than who did. This matters because a naive model could appear strong simply by predicting the majority class. I would keep this imbalance in mind and use class-aware evaluation in any later modeling stage.

13. Age is not evenly distributed, and the mean and median values are not identical. This suggests skew and possible outlier influence. I would report the distribution clearly and consider whether a log or robust scaling approach is needed later, but I would not alter the raw data during EDA.

14. Fare is strongly skewed and has a long tail, which means a few expensive tickets may dominate comparisons if raw values are used. This matters for any model that relies on fare as a predictor. I would keep this pattern visible in the analysis and consider transformations only in a later preprocessing workflow.

## Relationships

15. Passenger class and sex appear to be strongly related to survival outcomes. This is the kind of relationship that usually matters in an exploratory analysis because it tells us where the real signal is likely to be. I would treat them as core features in any future model, while still checking for confounding and leakage.

16. The correlation heatmap for the numeric features shows that the strongest relationships are concentrated in a small subset of variables rather than spread evenly across the table. This matters because it tells us where to focus attention for later modeling and where the feature space is likely to be noisy or weak. I would treat the heatmap as a guide for feature prioritisation, not as proof of causal effect.

17. Passenger class and sex show a clear relationship with survival, which is visible in the count plots and in the patterns inside the correlation matrix. This is a strong signal and likely one of the most informative predictors in the dataset. I would keep these variables high on the list of features to investigate in any future model.

18. Age and fare have visible relationship patterns with survival, but the shapes are not purely linear. This matters because a simple correlation statistic may understate how useful those features are. I would not rely on Pearson correlation alone when the relationship is likely to be nonlinear or confounded by class and sex.

19. The age-versus-fare scatter plot reveals that some relationships are difficult to interpret in a single linear summary. A variable can show a weak overall correlation while still containing useful structure when combined with another feature. I would treat correlation as one piece of evidence rather than the whole story.

## Risks and next steps

20. One material risk is that the dataset does not provide a strong time dimension. Without a timestamp or a train/test time split, it is easy to accidentally evaluate a model in a way that is optimistic and unrealistic. I would insist on a clear validation plan before any modeling begins.

21. Another risk is that missingness is not random. Age and Cabin are missing in structured ways, and later model performance may be distorted if those patterns correlate with passenger type or ticket information. I would record the missingness patterns and treat them as a core problem to solve in the preparation stage.

22. A final risk is that the target could be influenced by variables that are not truly available at prediction time. For example, features related to post-incident outcome or passenger status would be invalid for a realistic decision-making system. I would explicitly check for leakage before any model training begins.

## Overall conclusion

This dataset is suitable for exploratory analysis because it is messy in exactly the way a real-world dataset should be: missing values, some categorical complexity, class imbalance, and a clear target. The EDA shows that there are real data-quality issues and meaningful relationships, but there is no evidence of a model-ready dataset yet. The next step would be careful preprocessing and validation, not model tuning.
