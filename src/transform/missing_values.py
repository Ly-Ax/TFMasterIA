"""Module to impute missing values"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


class ModeImputer(BaseEstimator, TransformerMixin):
    """Univariate imputer with mode"""

    def __init__(self, variables):
        """Initialize variables"""
        self.variables = variables

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        for var in self.variables:
            col_values = X[[var]]
            mode_imputer = SimpleImputer(strategy="most_frequent")
            X[var] = mode_imputer.fit_transform(col_values)[:, 0]
        return X


class ClassImputer(BaseEstimator, TransformerMixin):
    """Multivariate imputer with RF Classifier"""

    def __init__(self, variables, df_subset):
        """Initialize variables"""
        self.variables = variables
        self.df_subset = df_subset

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        for var in self.variables:
            if X[var].isnull().sum() > 0:
                self.df_subset[0] = var
                df_ = X[self.df_subset]

                df_train = df_.dropna(subset=[var])
                df_test = df_[df_[var].isnull()]

                X_train = df_train.drop(columns=[var])
                y_train = df_train[var]
                X_test = df_test.drop(columns=[var])

                rf_classifier = RandomForestClassifier()
                rf_classifier.fit(X_train, y_train)

                y_test = rf_classifier.predict(X_test)
                X.loc[X[var].isnull(), var] = y_test

                X[var] = X[var].astype(int)
        return X


class DropDuplicates(BaseEstimator, TransformerMixin):
    """Drop duplicate values"""

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        X.drop_duplicates(inplace=True)
        return X
