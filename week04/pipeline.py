import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge

standard_columns = [
    "latitude",
    "longitude"
]

log_columns = [
    "minimum_nights",
    "number_of_reviews"
]

log_robust_columns = [
    "calculated_host_listings_count"
]

availability_columns = [
    "availability_365"
]

low_card_columns = [
    "neighbourhood_group",
    "room_type"
]

high_card_columns = [
    "neighbourhood"
]

standard_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scaler", StandardScaler())
])

log_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler())
])

robust_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", RobustScaler())
])

availability_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scaler", MinMaxScaler())
])

review_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True, keep_empty_features=True)),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler())
])

#Create the categorical transformers

low_card_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encoder",  OneHotEncoder(handle_unknown="ignore"))
])

high_card_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encoder", OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist"))
])

def build_preprocessor():

    preprocessor = ColumnTransformer(
            transformers=[
                ("standard", standard_pipeline, standard_columns),
                ("log", log_pipeline, log_columns),
                ("robust", robust_pipeline, log_robust_columns),
                ("availability", availability_pipeline, availability_columns),
                ("reviews", review_pipeline, ["reviews_per_month"]),
                ("low_card", low_card_pipeline, low_card_columns),
                ("high_card", high_card_pipeline, high_card_columns)
            ],
            remainder="drop"
    )

    return preprocessor

def build_pipeline():

    model = Pipeline([("pre", build_preprocessor()),
                      ("model", Ridge(alpha=10.0))
                      ])

    return model