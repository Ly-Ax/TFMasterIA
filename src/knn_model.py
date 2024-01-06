import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from classifier import classifier_models as cls_mod

test_data = cls_mod.GenerateTestTrain()
X_test, y_test = test_data.SampleData(100)

X = X_test[y_test == 1].sample(1)
dic_test = X.to_dict(orient="records")
print(dic_test)    

X = pd.DataFrame(dic_test)

knn_model = cls_mod.KnnModel()
y_pred = knn_model.KnnPredict(X)
print(y_pred)
