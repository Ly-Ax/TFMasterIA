import pandas as pd
import os
import streamlit as st
from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from classifier import classifier_models as cls_mod

def model_predict(model_name, input_df):
    if model_name == 'LR': 
        model = cls_mod.LogRegModel()
        y_pred = model.LogRegPredict(input_df)

    elif model_name == 'KNN': 
        model = cls_mod.KnnModel()
        y_pred = model.KnnPredict(input_df)
    
    elif model_name == 'DT': 
        model = cls_mod.DecTreeModel()
        y_pred = model.DecTreePredict(input_df)

    elif model_name == 'RF': 
        model = cls_mod.RanForModel()
        y_pred = model.RanForPredict(input_df)

    elif model_name == 'XGB': 
        model = cls_mod.XGBoostModel()
        y_pred = model.XGBoostPredict(input_df)

    return y_pred

def main_run():
    image = Image.open(os.getcwd() + '/images/viu_cabecera.webp')
    st.sidebar.image(image)
    add_selectbox = st.sidebar.selectbox('Select a Model:', 
                                         ('Logistic Regression',
                                          'K Neighbors',
                                          'Decision Tree',
                                          'Random Forest', 
                                          'XGBoost'))
    st.sidebar.info('The code is located in the repository:')
    st.sidebar.success('https://github.com/Ly-Ax/TFMasterIA')

    st.title("Bank Loan Default Prediction")

    lst_state = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
    state = st.selectbox('State:', lst_state)
    lst_bank = ['AK', 'AL', 'AN', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'EN', 'FL', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VI', 'VT', 'WA', 'WI', 'WV', 'WY']
    bank_state = st.selectbox('Bank State:', lst_bank)
    lst_naics = ['0', '11', '21', '22', '23', '31', '32', '33', '42', '44', '45', '48', '49', '51', '52', '53', '54', '55', '56', '61', '62', '71', '72', '81', '92']
    naics = st.selectbox('NAICS:', lst_naics)
    approval_date = st.text_input('Approval Date:', value='13-Jan-06', max_chars=9)
    term = st.number_input('Term:', min_value=0, max_value=569, value=0)
    no_emp = st.number_input('No Emp:', min_value=0, max_value=9999, value=0)
    new_exist = st.selectbox('New Exist:', [1.0, 2.0]) 
    urban_rural= st.selectbox('Urban Rural:', [0, 1, 2])
    rev_line = st.selectbox('Rev Line:', ['1', '0'])
    low_doc = st.selectbox('Low Doc:', ['Y', 'N'])
    disbursement_gross = st.text_input('Disbursement Gross:', value='$100.00')
    gr_appv = st.text_input('Gross Approv:', value='$100.00')
    sba_appv = st.text_input('SBA Approv:', value='$70.00')

    input_dict = {'LoanNr_ChkDgt': 0, 
                  'Name': '', 
                  'City': '', 
                  'State': state, 
                  'Zip': 0, 
                  'Bank': '', 
                  'BankState': bank_state, 
                  'NAICS': naics, 
                  'ApprovalDate': approval_date, 
                  'ApprovalFY': '', 
                  'Term': term, 
                  'NoEmp': no_emp, 
                  'NewExist': new_exist, 
                  'CreateJob': 0, 
                  'RetainedJob': 0, 
                  'FranchiseCode': 0, 
                  'UrbanRural': urban_rural, 
                  'RevLineCr': rev_line, 
                  'LowDoc': low_doc, 
                  'ChgOffDate': '', 
                  'DisbursementDate': '', 
                  'DisbursementGross': disbursement_gross, 
                  'BalanceGross': '', 
                  'ChgOffPrinGr': '', 
                  'GrAppv': gr_appv, 
                  'SBA_Appv': sba_appv}
    output = ''
    input_df = pd.DataFrame([input_dict])

    if add_selectbox == 'Logistic Regression':
        if st.button('Predict'):
            output = model_predict('LR', input_df)
    
    elif add_selectbox == 'K Neighbors':
        if st.button('Predict'):
            output = model_predict('KNN', input_df)
    
    elif add_selectbox == 'Decision Tree':
        if st.button('Predict'):
            output = model_predict('DT', input_df)
    
    elif add_selectbox == 'Random Forest':
        if st.button('Predict'):
            output = model_predict('RF', input_df)
    
    elif add_selectbox == 'XGBoost':
        if st.button('Predict'):
            output = model_predict('XGB', input_df)

    st.success(f'Default: {output}')

if __name__ == '__main__':
    main_run()

# streamlit run c:/Ly-Ax/TFMasterIA/apps/predict_data.py
