"""Encode categorical features"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


class LabelTransformer(BaseEstimator, TransformerMixin):
    """Encode categorical nominal features"""

    def __init__(self, variables):
        """Initialize variables"""
        self.variables = variables
        self.encoders = {}

    def fit(self, X, y=None):
        """Fit custom transformer"""
        for var in self.variables:
            encoder = LabelEncoder()
            encoder.fit(X[var])
            self.encoders[var] = encoder
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        for var, encoder in self.encoders.items():
            X[var] = encoder.transform(X[var])
        return X


class OneHotTransformer(BaseEstimator, TransformerMixin):
    """Encode categorical numerical features"""

    def __init__(self, variable):
        """Initialize variables"""
        self.variable = variable

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        X = pd.get_dummies(X, columns=[self.variable])

        X.rename(
            columns={"UrbanRural_1": "Urban", "UrbanRural_2": "Rural"}, inplace=True
        )
        X.drop(columns="UrbanRural_0", axis=1, inplace=True)

        X["Urban"] = X["Urban"].astype(int)
        X["Rural"] = X["Rural"].astype(int)
        return X


class OrdinalTransformer(BaseEstimator, TransformerMixin):
    """Encode categorical ordinal features"""

    def __init__(self, variable):
        """Initialize variables"""
        self.variable = variable
        self.encoder = OrdinalEncoder()

    def fit(self, X, y=None):
        """Fit custom transformer"""
        self.encoder.categories_ = list(sorted(X[self.variable].unique()))
        self.encoder.fit(X[[self.variable]])
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        var = self.variable

        X[var] = self.encoder.transform(X[[var]])
        X[var] = X[var].astype(int)
        return X


class SortColumns(BaseEstimator, TransformerMixin):
    """Sort columns of final dataset"""

    def __init__(self, variables):
        """Initialize variables"""
        self.variables = variables

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        X = X[self.variables]
        return X
