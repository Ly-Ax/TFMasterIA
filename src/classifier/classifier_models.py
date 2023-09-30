"""Classifier models"""
import pandas as pd
import numpy as np
import os
import sys
import yaml
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).parents[1]))

import transform_main as trn_main
from classifier import custom_classifier as cs_cl


class GenerateTestTrain:
    """Generate test data"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.target = "Default"

    def SampleData(self, NumSample=0):
        """Generate sample data"""
        df_test = pd.read_csv(
            self.path + self.config["data"]["data_test"], low_memory=False
        )
        if NumSample > 0:
            df_test = df_test.sample(NumSample)

        df_test["MIS_Status"] = np.where(
            df_test["MIS_Status"] == "CHGOFF", 1, df_test["MIS_Status"]
        )
        df_test["MIS_Status"] = np.where(
            df_test["MIS_Status"] == "P I F", 0, df_test["MIS_Status"]
        )
        df_test["MIS_Status"] = df_test["MIS_Status"].astype("Int64")
        df_test.rename(columns={"MIS_Status": "Default"}, inplace=True)

        df_test.dropna(subset=["Default"], inplace=True)
        df_test["Default"] = df_test["Default"].astype(int)

        df_test.drop_duplicates(inplace=True)

        X_test = df_test.drop(columns=["Default"])
        y_test = df_test["Default"]

        return X_test, y_test

    def TrainData(self, ValData=False):
        """Generate train data"""
        trn_data = trn_main.TransformData()

        df_train = pd.read_csv(
            self.path + self.config["data"]["data_train"], low_memory=False
        )
        df_train = trn_data.Preprocessing(df_train)

        if ValData == True:
            df_val = pd.read_csv(
                self.path + self.config["data"]["data_val"], low_memory=False
            )
            df_val = trn_data.Preprocessing(df_val)
            df_train = pd.concat([df_train, df_val], axis=0)

        X = df_train.drop(columns=[self.target])
        y = df_train[self.target]

        return X, y


class LogRegModel:
    """Logistic Regression model"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.num_cols = [
            "Term",
            "NoEmp",
            "SecuredSBA",
            "GrDisburs",
            "GrApprov",
            "ApprovSBA",
        ]

    def TrainLogReg(self):
        """Train Logistic Regression model"""
        train_data = GenerateTestTrain()
        X_train, y_train = train_data.TrainData()

        logreg_pipeline = Pipeline(
            [
                ("scaler", cs_cl.ZScoreTransformer(self.num_cols)),
                ("custom_model", cs_cl.LogisticRegressionModel()),
            ]
        )
        logreg_pipeline.fit(X_train, y_train)

        joblib.dump(logreg_pipeline, self.path + self.config["models"]["logreg_model"])

    def LogRegPredict(self, X):
        """Logistic Regression Predict"""
        trn_data = trn_main.TransformData()
        X = trn_data.Preprocessing(X)

        logreg_model = joblib.load(self.path + self.config["models"]["logreg_model"])

        y_pred = logreg_model.predict(X)
        return y_pred


class KnnModel:
    """K Nearest Neighbors model"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.num_cols = [
            "Term",
            "NoEmp",
            "SecuredSBA",
            "GrDisburs",
            "GrApprov",
            "ApprovSBA",
        ]

    def TrainKnn(self):
        """Train K Nearest Neighbors model"""
        train_data = GenerateTestTrain()
        X_train, y_train = train_data.TrainData(ValData=True)

        knn_pipeline = Pipeline(
            [
                ("scaler", cs_cl.MinMaxTransformer(self.num_cols)),
                ("custom_model", cs_cl.KNeighborsModel()),
            ]
        )
        knn_pipeline.fit(X_train, y_train)

        joblib.dump(knn_pipeline, self.path + self.config["models"]["knn_model"])

    def KnnPredict(self, X):
        """K Nearest Neighbors Predict"""
        trn_data = trn_main.TransformData()
        X = trn_data.Preprocessing(X)

        knn_model = joblib.load(self.path + self.config["models"]["knn_model"])

        y_pred = knn_model.predict(X)
        return y_pred


class DecTreeModel:
    """Decision Tree Classifier"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

    def TrainDecTree(self):
        """Train Decision Tree Classifier"""
        train_data = GenerateTestTrain()
        X_train, y_train = train_data.TrainData(ValData=True)

        dectree_pipeline = Pipeline([("custom_model", cs_cl.DecisionTreeModel())])
        dectree_pipeline.fit(X_train, y_train)

        joblib.dump(
            dectree_pipeline, self.path + self.config["models"]["dectree_model"]
        )

    def DecTreePredict(self, X):
        """Decision Tree Classifier Predict"""
        trn_data = trn_main.TransformData()
        X = trn_data.Preprocessing(X)

        dtc_model = joblib.load(self.path + self.config["models"]["dectree_model"])

        y_pred = dtc_model.predict(X)
        return y_pred


class RanForModel:
    """Random Forest Classifier"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

    def TrainRanFor(self):
        """Train Random Forest Classifier"""
        train_data = GenerateTestTrain()
        X_train, y_train = train_data.TrainData(ValData=True)

        ranfor_pipeline = Pipeline([("custom_model", cs_cl.RandomForestModel())])
        ranfor_pipeline.fit(X_train, y_train)

        joblib.dump(ranfor_pipeline, self.path + self.config["models"]["ranfor_model"])

    def RanForPredict(self, X):
        """Random Forest Classifier Predict"""
        trn_data = trn_main.TransformData()
        X = trn_data.Preprocessing(X)

        rfc_model = joblib.load(self.path + self.config["models"]["ranfor_model"])

        y_pred = rfc_model.predict(X)
        return y_pred


class XGBoostModel:
    """XGBoost Classifier"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

    def TrainXGBoost(self):
        """Train Random Forest Classifier"""
        train_data = GenerateTestTrain()
        X_train, y_train = train_data.TrainData(ValData=True)

        xgboost_pipeline = Pipeline([("custom_model", cs_cl.XGBoostModel())])
        xgboost_pipeline.fit(X_train, y_train)

        joblib.dump(
            xgboost_pipeline, self.path + self.config["models"]["xgboost_model"]
        )

    def XGBoostPredict(self, X):
        """XGBoost Classifier Predict"""
        trn_data = trn_main.TransformData()
        X = trn_data.Preprocessing(X)

        xgb_model = joblib.load(self.path + self.config["models"]["xgboost_model"])

        y_pred = xgb_model.predict(X)
        return y_pred


if __name__ == "__main__":
    try:
        test_data = GenerateTestTrain()
        X_test, y_test = test_data.SampleData(100)
        # print(f"X: {X_test.shape} y: {y_test.shape}")
        # X_train, y_train = test_data.TrainData(ValData=True)
        # print(f"X: {X_train.shape} y: {y_train.shape}")

        # lr_model = LogRegModel()
        # # lr_model.TrainLogReg()
        # y_pred = lr_model.LogRegPredict(X_test)

        # knn_model = KnnModel()
        # # knn_model.TrainKnn()
        # y_pred = knn_model.KnnPredict(X_test)

        # dtc_model = DecTreeModel()
        # # dtc_model.TrainDecTree()
        # y_pred = dtc_model.DecTreePredict(X_test)

        # rfc_model = RanForModel()
        # # rfc_model.TrainRanFor()
        # y_pred = rfc_model.RanForPredict(X_test)

        xgb_model = XGBoostModel()
        # xgb_model.TrainXGBoost()
        y_pred = xgb_model.XGBoostPredict(X_test)

        print("Exactitud:    %.4f" % (accuracy_score(y_test, y_pred)))
        print("Precisión:    %.4f" % (precision_score(y_test, y_pred, average="macro")))
        print("Sensibilidad: %.4f" % (recall_score(y_test, y_pred, average="macro")))
        print("F1-score:     %.4f" % (f1_score(y_test, y_pred, average="macro")))

    except Exception as err:
        print("Error: ", str(err))
