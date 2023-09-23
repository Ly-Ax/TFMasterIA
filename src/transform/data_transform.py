"""Module to transform data features"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


"""==================== FEATURE TRANSFORM ===================="""


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

        if var in X.columns:
            X[var] = np.where(X[var] == "CHGOFF", 1, X[var])
            X[var] = np.where(X[var] == "P I F", 0, X[var])
            X[var] = X[var].astype("Int64")

            X.rename(columns={var: "Default"}, inplace=True)
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


"""==================== MISSING VALUES ===================="""


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

    def __init__(self, variable):
        """Initialize variables"""
        self.variable = variable

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        if self.variable[0] in X.columns:
            X = X.copy()
            X.drop_duplicates(inplace=True)
        return X


"""==================== ENCODE FEATURES ===================="""


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
        # X = pd.get_dummies(X, columns=[self.variable])
        # X.rename(columns={"UrbanRural_1":"Urban",
        #                   "UrbanRural_2":"Rural"}, inplace=True)
        # X.drop(columns="UrbanRural_0", axis=1, inplace=True)
        X["Urban"] = np.where(X["UrbanRural"]==1, 1, 0)
        X["Rural"] = np.where(X["UrbanRural"]==2, 1, 0)
        X.drop(columns="UrbanRural", axis=1, inplace=True)

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


class DropNaNs(BaseEstimator, TransformerMixin):
    """Drop null values"""

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
            if var in X.columns:
                X.dropna(subset=[var], inplace=True)
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
        sort_cols = [col for col in self.variables if col in X.columns]
        X = X[sort_cols]
        return X


"""==================== RESAMPLING TRAIN ===================="""


class SubSampling(BaseEstimator, TransformerMixin):
    """Undersampling of training data"""

    def __init__(self, target, rand_seed):
        """Initialize variables"""
        self.target = target
        self.rand_seed = rand_seed

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        y = X[self.target]
        X = X.drop(columns=self.target)

        undersampler = RandomUnderSampler(
            sampling_strategy="auto", random_state=self.rand_seed
        )
        X_resampled, y_resampled = undersampler.fit_resample(X, y)

        df_under = pd.DataFrame(X_resampled)
        df_under[self.target] = y_resampled
        return df_under


class SmoteSampling(BaseEstimator, TransformerMixin):
    """Oversampling of training data with SMOTE"""

    def __init__(self, target, rand_seed):
        """Initialize variables"""
        self.target = target
        self.rand_seed = rand_seed

    def fit(self, X, y=None):
        """Fit custom transformer"""
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        y = X[self.target]
        X = X.drop(columns=self.target)

        smote = SMOTE(sampling_strategy="auto", random_state=self.rand_seed)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        df_smote = pd.DataFrame(X_resampled)
        df_smote[self.target] = y_resampled
        return df_smote
