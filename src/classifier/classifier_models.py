import pandas as pd
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


class GenerateTestData:
    def __init__(self, num_sample=0):
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        
        self.sample = num_sample
    
    def SampleData(self):
        df_test = pd.read_csv(self.path + self.config["data"]["data_test"], low_memory=False)
        if self.sample != 0: df_test = df_test.sample(self.sample)

        X_test = df_test.drop(columns=["MIS_Status"])
        y_test = df_test["MIS_Status"]

        return X_test, y_test
    

class LogRegModel:
    def __init__(self):
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
        self.target = ["Default"]
    
    def TrainLogReg(self, SaveModel=False):
        trn_data = trn_main.TransformData()
        df_train = trn_data.PreproPipeline(self.config["data"]["data_train"])
        df_val = trn_data.PreproPipeline(self.config["data"]["data_val"])
        
        df_train = pd.concat([df_train, df_val], axis=0)
        X = df_train.drop(columns=self.target)
        y = df_train[self.target[0]]

        logreg_pipeline = Pipeline(
            [
                ("scaler", cs_cl.ZScoreTransformer(self.num_cols)),
                ("custom_model", cs_cl.LogisticRegressionModel()),
            ]
        )
        logreg_pipeline.fit_transform(X, y)

        if SaveModel == True:
            joblib.dump(logreg_pipeline, self.path + self.config["models"]["logreg_model"])

        return logreg_pipeline

    def LogRegPredict(self, X):
        trn_data = trn_main.TransformData()
        X_test = trn_data.Preprocessing(X)

        logreg_model = joblib.load(self.path + self.config["models"]["logreg_model"])
        
        y_pred = logreg_model.predict(X_test)
        return y_pred

if __name__ == "__main__":
    try:
        test_data = GenerateTestData()
        X_test, y_test = test_data.SampleData()

        lr_model = LogRegModel()
        # log_reg = lr_model.TrainLogReg(SaveModel=False)
        y_pred = lr_model.LogRegPredict(X_test)
        
        print(y_pred.shape)
        # print("Exactitud:    %.4f" % (accuracy_score(y_test, y_pred)))
        # print("Precisión:    %.4f" % (precision_score(y_test, y_pred, average="macro")))
        # print("Sensibilidad: %.4f" % (recall_score(y_test, y_pred, average="macro")))
        # print("F1-score:     %.4f" % (f1_score(y_test, y_pred, average="macro")))

    except Exception as err:
        print("Error: ", str(err))