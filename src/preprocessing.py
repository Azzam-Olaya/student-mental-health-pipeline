import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class IQRClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        self.low_ = q1 - 1.5 * (q3 - q1)
        self.high_ = q3 + 1.5 * (q3 - q1)
        return self

    def transform(self, X):
        return np.clip(X, self.low_, self.high_)


def build_preprocessor(x_train):
    num_cols = x_train.select_dtypes(include="number").columns.tolist()
    onehot_cols = ["Gender", "Part_Time_Job", "Previous_Mental_Health_History"]
    ordinal_cols = ["Physical_Activity", "Diet_Quality"]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outliers", IQRClipper()),
        ("scaler", StandardScaler()),
    ])
    onehot_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["Low", "Moderate", "High"], ["Poor", "Average", "Good"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("onehot", onehot_pipe, onehot_cols),
        ("ordinal", ordinal_pipe, ordinal_cols),
    ])
