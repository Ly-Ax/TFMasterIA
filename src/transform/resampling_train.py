"""Data resampling for training"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


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
