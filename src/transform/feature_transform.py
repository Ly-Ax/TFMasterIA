"""Module to transform features"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumns(BaseEstimator, TransformerMixin):
    """Remove specific columns"""

    def __init__(self, variables):
        """Initialize variables"""
        self.variables = variables

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        X = X.drop(columns=self.variables)
        return X


class CurrencyToInt(BaseEstimator, TransformerMixin):
    """Convert currency to int"""

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
            X[var] = X[var].str.replace(r"[\$,]", "", regex=True).astype(float)
            X[var] = X[var].round().astype(int)
        return X


class ApprovalDate(BaseEstimator, TransformerMixin):
    """Format approval date"""

    def __init__(self, variable):
        """Initialize variables"""
        self.variable = variable

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def __format_date(self, app_date):
        """Format date variable"""
        date = app_date.split("-")
        year = date[2]
        if int(year) > 14:
            year = "19" + year
        else:
            year = "20" + year
        return f"{date[0]}-{date[1]}-{year}"

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        var = self.variable
        var_year = var[0:3] + "Year"
        var_month = var[0:3] + "Month"

        X[var] = pd.to_datetime(X[var].apply(self.__format_date))
        X[var_year] = X[var].dt.year
        X[var_month] = X[var].dt.month
        X = X.drop(columns=var)
        return X


class CategorizeNAICS(BaseEstimator, TransformerMixin):
    """Extract Sector from NAICS"""

    def __init__(self, variable, sector):
        """Initialize variables"""
        self.variable = variable
        self.sector = sector

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        var = self.variable

        X[var] = X[var].astype(str).str[0:2]
        X[var] = X[var].apply(lambda x: self.sector[x])
        return X


class ConvertNewExist(BaseEstimator, TransformerMixin):
    """Convert NewExist column"""

    def __init__(self, variable):
        """Initialize variables"""
        self.variable = variable

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        var = self.variable

        X[var] = np.where(X[var] == 0.0, np.nan, X[var])
        X[var] = np.where(X[var] == 1, 0, X[var])
        X[var] = np.where(X[var] == 2, 1, X[var])
        X[var] = X[var].astype("Int64")
        return X


class ConvertRevLineCr(BaseEstimator, TransformerMixin):
    """Convert RevLineCr column"""

    def __init__(self, variable):
        """Initialize variables"""
        self.variable = variable

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        var = self.variable

        X[var] = np.where(X[var].isin(["Y", "T"]), "1", X[var])
        X[var] = np.where(X[var].isin(["N"]), "0", X[var])
        X[var] = np.where(~X[var].isin(["1", "0"]), np.nan, X[var])
        X[var] = X[var].astype("Int64")
        return X


class ConvertLowDoc(BaseEstimator, TransformerMixin):
    """Convert LowDoc column"""

    def __init__(self, variable):
        """Initialize variables"""
        self.variable = variable

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        var = self.variable

        X[var] = np.where(X[var] == "Y", "1", X[var])
        X[var] = np.where(X[var] == "N", "0", X[var])
        X[var] = np.where(~X[var].isin(["1", "0"]), np.nan, X[var])
        X[var] = X[var].astype("Int64")
        return X


class ConvertTarget(BaseEstimator, TransformerMixin):
    """Convert target column"""

    def __init__(self, variable):
        """Initialize variables"""
        self.variable = variable

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        var = self.variable

        X[var] = np.where(X[var] == "CHGOFF", 1, X[var])
        X[var] = np.where(X[var] == "P I F", 0, X[var])
        X[var] = X[var].astype("Int64")
        return X


class CreateFeatures(BaseEstimator, TransformerMixin):
    """Create new features"""

    def __init__(self, variables):
        """Initialize variables"""
        self.variables = variables

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        vars = self.variables

        X[vars[0]] = np.where(X["State"] != X["BankState"], 1, 0)
        X[vars[1]] = np.where(X["Term"] >= 240, 1, 0)
        X[vars[2]] = round((X["SBA_Appv"] / X["GrAppv"]) * 100)
        X[vars[2]] = X[vars[2]].astype(int)
        return X


class RenameColumns(BaseEstimator, TransformerMixin):
    """Rename existing columns"""

    def __init__(self, variables):
        """Initialize variables"""
        self.variables = variables

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        X.rename(columns=self.variables, inplace=True)
        return X
