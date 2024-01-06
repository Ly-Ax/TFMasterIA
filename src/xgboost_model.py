import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from classifier import classifier_models as cls_mod

class SmallBA(BaseModel):
    LoanNr_ChkDgt: object
    Name: object 
    City: object
    State: object
    Zip: int
    Bank: object
    BankState: object
    NAICS: int
    ApprovalDate: object
    ApprovalFY: object
    Term: int
    NoEmp: int 
    NewExist: float
    CreateJob: int 
    RetainedJob: int 
    FranchiseCode: int
    UrbanRural: int
    RevLineCr: object
    LowDoc: object
    ChgOffDate: object
    DisbursementDate: object
    DisbursementGross: object
    BalanceGross: object 
    ChgOffPrinGr: object
    GrAppv: object
    SBA_Appv: object

app = FastAPI()
model = cls_mod.XGBoostModel()

@app.get("/")
def home():
    return 'Classifier is ready...'

@app.post("/predict")
def predict_xgboost(client:SmallBA):
    X = dict(client)
    X = pd.DataFrame([X])
    pred = model.XGBoostPredict(X)

    return {'Prediction': str(pred[0])}

# 1: {"LoanNr_ChkDgt": 9570974007, "Name": "NEW HORIZONS SOLUTIONS INC", "City": "BRENTWOOD", "State": "TN", "Zip": 37027, "Bank": "BBCN BANK", "BankState": "CA", "NAICS": 541612, "ApprovalDate": "4-Jan-06", "ApprovalFY": "2006", "Term": 61, "NoEmp": 1, "NewExist": 1.0, "CreateJob": 2, "RetainedJob": 1, "FranchiseCode": 1, "UrbanRural": 1, "RevLineCr": "0", "LowDoc": "N", "ChgOffDate": "9-Mar-08", "DisbursementDate": "31-Jan-06", "DisbursementGross": "$25,000.00 ", "BalanceGross": "$0.00 ", "ChgOffPrinGr": "$23,648.00 ", "GrAppv": "$25,000.00 ", "SBA_Appv": "$21,250.00 "}
# 0: {"LoanNr_ChkDgt": 8002113008, "Name": "BIRMINGHAMS FLOWERS & GREENHOU", "City": "STURGEON BAY", "State": "WI", "Zip": 54235, "Bank": "BAYLAKE BANK", "BankState": "WI", "NAICS": 453110, "ApprovalDate": "23-Jan-95", "ApprovalFY": "1995", "Term": 180, "NoEmp": 5, "NewExist": 1.0, "CreateJob": 0, "RetainedJob": 0, "FranchiseCode": 1, "UrbanRural": 0, "RevLineCr": "N", "LowDoc": "N", "ChgOffDate": "nan", "DisbursementDate": "30-Apr-95", "DisbursementGross": "$130,000.00 ", "BalanceGross": "$0.00 ", "ChgOffPrinGr": "$0.00 ", "GrAppv": "$130,000.00 ", "SBA_Appv": "$110,500.00 "}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port='8000') # http://127.0.0.1:8000
