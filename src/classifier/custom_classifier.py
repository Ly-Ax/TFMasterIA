"""Transformations and custom models for classification"""
import os
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin


class ZScoreTransformer(BaseEstimator, TransformerMixin):
    """Normalize numerical features"""

    def __init__(self, variables):
        """Initialize variables"""
        self.variables = variables
        self.encoders = {}

    def fit(self, X, y=None):
        """Fit custom transformer"""
        for var in self.variables:
            encoder = StandardScaler()
            encoder.fit(X[var].values.reshape(-1, 1))
            self.encoders[var] = encoder
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        for var, encoder in self.encoders.items():
            X[var] = encoder.transform(X[var].values.reshape(-1, 1))
        return X


class LogisticRegressionModel(BaseEstimator, TransformerMixin):
    """Logistic Regression Model"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.log_reg = LogisticRegression(
            penalty = self.config["logreg"]["penalty"],
            C = self.config["logreg"]["c"],
            max_iter = self.config["logreg"]["max_iter"],
            random_state = self.config["vars"]["rand_logreg"],
        )
        self.threshold = self.config["logreg"]["threshold"]

    def fit(self, X, y):
        """Fit logistic regression model"""
        self.log_reg.fit(X, y)
        return self

    def transform(self, X):
        """Apply transformation"""
        return X

    def predict(self, X, y=None):
        """Custom predictive logistic regression model"""
        y_prob = self.log_reg.predict_proba(X)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        return y_pred
