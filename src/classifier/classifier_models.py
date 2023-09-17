import pandas as pd
import os
import sys
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from classifier import logreg_pipeline as lr_pipe

class GenerateTest:
    def __init__(self, num_sample):
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        
        self.sample = num_sample
    
    def ExtractData(self):
        df_test = pd.read_csv(self.path + self.config["data"]["data_test"])
        df_test = df_test.sample(self.sample)

        X_test = df_test.drop(columns=self.target)
        y_test = df_test[[self.target[0]]]

        return X_test, y_test

class LogRegClassifier:

    def __init__(self):
        pass

    def PredictModel(self, X_test):
        # clean X_test (preprocessing)
        lr_train = lr_pipe.LogRegPipeline()
        lr_model = lr_train.LogRegModel()

        y_pred = lr_model.predict(X_test)
        return y_pred
    
    def SaveModel(self):
        pass

if __name__ == "__main__":
    try:
        test = GenerateTest(10)
        X_test, y_test = test.ExtractData()
        
        # model = LogRegClassifier()
        # y_pred = model.LogRegClassifier(X_test)

        # print(y_pred)

    except Exception as err:
        print("Error: ", str(err))