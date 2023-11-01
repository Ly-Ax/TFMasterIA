"""Transformations and custom models for classification"""
import os
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


"""==================== TRANSFORMATIONS ===================="""


class ZScoreTransformer(BaseEstimator, TransformerMixin):
    """Standardize numerical features"""

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


class MinMaxTransformer(BaseEstimator, TransformerMixin):
    """Normalize numerical features"""

    def __init__(self, variables):
        """Initialize variables"""
        self.variables = variables
        self.encoders = {}

    def fit(self, X, y=None):
        """Fit custom transformer"""
        for var in self.variables:
            encoder = MinMaxScaler()
            encoder.fit(X[var].values.reshape(-1, 1))
            self.encoders[var] = encoder
        return self

    def transform(self, X, y=None):
        """Apply transformation"""
        X = X.copy()
        for var, encoder in self.encoders.items():
            X[var] = encoder.transform(X[var].values.reshape(-1, 1))
        return X


"""==================== LOGISTIC REGRESSION ===================="""


class LogisticRegressionModel(BaseEstimator, TransformerMixin):
    """Logistic Regression Model"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.log_reg = LogisticRegression(
            penalty=self.config["logreg"]["penalty"],
            C=self.config["logreg"]["c"],
            max_iter=self.config["logreg"]["max_iter"],
            random_state=self.config["vars"]["rand_logreg"],
        )
        self.threshold = self.config["logreg"]["threshold"]

    def fit(self, X, y):
        """Fit logistic regression model"""
        self.log_reg.fit(X, y)
        return self

    def predict(self, X, y=None):
        """Custom predictive logistic regression model"""
        y_prob = self.log_reg.predict_proba(X)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        return y_pred


"""==================== K NEAREST NEIGHBORS ===================="""


class KNeighborsModel(BaseEstimator, TransformerMixin):
    """K Nearest Neighbors Model"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.knn = KNeighborsClassifier(
            n_neighbors=self.config["knn"]["n_neighbors"],
            weights=self.config["knn"]["weights"],
            metric=self.config["knn"]["metric"],
        )
        self.pred_cols = [
            "State",
            "DifState",
            "Sector",
            "AppYear",
            "Term",
            "Secured",
            "Urban",
            "RevLine",
            "LowDoc",
            "SecuredSBA",
        ]

    def fit(self, X, y):
        """Fit k nearest neighbors model"""
        self.knn.fit(X[self.pred_cols], y)
        return self

    def predict(self, X, y=None):
        """Custom predictive k nearest neighbors model"""
        X_ = X[self.pred_cols].copy()
        y_pred = self.knn.predict(X_)
        return y_pred


"""==================== DECISION TREE CLASSIFIER ===================="""


class DecisionTreeModel(BaseEstimator, TransformerMixin):
    """Decision Tree Classifier Model"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.dectree = DecisionTreeClassifier(
            criterion=self.config["dectree"]["criterion"],
            max_depth=self.config["dectree"]["max_depth"],
            random_state=self.config["vars"]["rand_dectree"],
        )
        self.pred_cols = [
            "State",
            "BankState",
            "DifState",
            "AppYear",
            "Term",
            "NoEmp",
            "GrDisburs",
            "ApprovSBA",
            "SecuredSBA",
        ]

    def fit(self, X, y):
        """Fit decision tree classifier"""
        self.dectree.fit(X[self.pred_cols], y)
        return self

    def predict(self, X, y=None):
        """Custom predictive decision tree classifier"""
        X_ = X[self.pred_cols].copy()
        y_pred = self.dectree.predict(X_)
        return y_pred


"""==================== RANDOM FOREST CLASSIFIER ===================="""


class RandomForestModel(BaseEstimator, TransformerMixin):
    """Random Forest Classifier Model"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.ranfor = RandomForestClassifier(
            criterion=self.config["ranfor"]["criterion"],
            max_depth=self.config["ranfor"]["max_depth"],
            random_state=self.config["vars"]["rand_ranfor"],
        )
        self.pred_cols = [
            "State",
            "BankState",
            "Sector",
            "AppYear",
            "Term",
            "GrDisburs",
            "GrApprov",
            "ApprovSBA",
            "SecuredSBA",
        ]

    def fit(self, X, y):
        """Fit random forest classifier"""
        self.ranfor.fit(X[self.pred_cols], y)
        return self

    def predict(self, X, y=None):
        """Custom predictive random forest classifier"""
        X_ = X[self.pred_cols].copy()
        y_pred = self.ranfor.predict(X_)
        return y_pred


"""==================== XGBOOST CLASSIFIER ===================="""


class XGBoostModel(BaseEstimator, TransformerMixin):
    """XGBoost Classifier Model"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.xgboost = XGBClassifier(
            learning_rate=self.config["xgboost"]["learning_rate"],
            max_depth=self.config["xgboost"]["max_depth"],
            n_estimators=self.config["xgboost"]["n_estimators"],
            random_state=self.config["vars"]["rand_xgboost"],
        )

    def fit(self, X, y):
        """Fit xgboost classifier"""
        self.xgboost.fit(X, y)
        return self

    def predict(self, X, y=None):
        """Custom predictive xgboost classifier"""
        y_pred = self.xgboost.predict(X)
        return y_pred
