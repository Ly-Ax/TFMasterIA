import pandas as pd
import os
import sys
import yaml
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).parents[1]))

from transform import data_pipelines as dt_pp
from classifier import custom_classifier as cs_cl


class LogRegPipeline:
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

    def LogRegModel(self):
        data_pipe = dt_pp.DataPipelines()
        df_train = data_pipe.PreprocessingPipeline(self.config["data"]["data_train"])
        df_val = data_pipe.PreprocessingPipeline(self.config["data"]["data_val"])
        
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

        return logreg_pipeline
    
    def LoadDataTest(self):
        df_test = pd.read_csv(self.path + self.config["data"]["clean_test"])

        X_test = df_test.drop(columns=self.target)
        y_test = df_test[[self.target[0]]]

        return X_test, y_test

if __name__ == "__main__":
    try:
        lr_train = LogRegPipeline()
        lr_model = lr_train.LogRegModel()

        X_test, y_test = lr_train.LoadDataTest()
        y_pred = lr_model.predict(X_test)

        print("Exactitud:    %.4f" % (accuracy_score(y_test, y_pred)))
        print("Precisión:    %.4f" % (precision_score(y_test, y_pred, average="macro")))
        print("Sensibilidad: %.4f" % (recall_score(y_test, y_pred, average="macro")))
        print("F1-score:     %.4f" % (f1_score(y_test, y_pred, average="macro")))

    except Exception as err:
        print("Error: ", str(err))