import pandas as pd
import os
import sys
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from classifier import classifier_models as lr_pipe

class LogRegClassifier:

    def __init__(self):
        pass

    def PredictModel(self, X_test):
        # clean X_test (preprocessing)
        lr_train = lr_pipe.LogRegPipeline()
        lr_model = lr_train.LogRegModel()

        y_pred = lr_model.predict(X_test)
        return y_pred
    

if __name__ == "__main__":
    try:
        # model = LogRegClassifier()
        # y_pred = model.LogRegClassifier(X_test)

        # print(y_pred)
        pass

    except Exception as err:
        print("Error: ", str(err))