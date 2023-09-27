import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).parents[1]))
from classifier import classifier_models as cls_mod

pred_mode = "batch"
test_data = cls_mod.GenerateTestData(100)
X_test, y_test = test_data.SampleData()

if pred_mode == "batch":
    knn_model = cls_mod.KnnModel()
    y_pred = knn_model.KnnPredict(X_test)

    print("Exactitud:    %.4f" % (accuracy_score(y_test, y_pred)))
    print("Precisión:    %.4f" % (precision_score(y_test, y_pred, average="macro")))
    print("Sensibilidad: %.4f" % (recall_score(y_test, y_pred, average="macro")))
    print("F1-score:     %.4f" % (f1_score(y_test, y_pred, average="macro")))

if pred_mode == "single":
    X = X_test[y_test == 1].sample(1)
    # dic_test = X.to_dict(orient="records")
    # print(dic_test)    
    # def_true = [{"State": 34, "BankState": 37, "DifState": 0, "Sector": 19, "AppYear": 43,
    #              "AppMonth": 7, "Term": 2, "NoEmp": 3, "Secured": 0, "NewExist": 0,
    #              "Urban": 1, "Rural": 0, "RevLine": 1, "LowDoc": 0, "GrDisburs": 189319,
    #              "GrApprov": 100000,"ApprovSBA": 50000,"SecuredSBA": 50,}]
    # def_false = [{"State": 19, "BankState": 22, "DifState": 0, "Sector": 1, "AppYear": 42, 
    #               "AppMonth": 6, "Term": 120, "NoEmp": 7, "Secured": 0, "NewExist": 0, 
    #               "Urban": 1, "Rural": 0, "RevLine": 1, "LowDoc": 0, "GrDisburs": 744915, 
    #               "GrApprov": 290000, "ApprovSBA": 145000, "SecuredSBA": 50,}]
    # X = pd.DataFrame(def_true)

    knn_model = cls_mod.KnnModel()
    y_pred = knn_model.KnnPredict(X)
    print(y_pred)
