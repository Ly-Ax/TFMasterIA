"""Main module from data pipelines"""
import pandas as pd
import os
import sys
import yaml
from pathlib import Path
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).parents[1]))
from transform import data_transform as dt_trn


class DataPipelines:
    """Pipelines to transform datasets"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.drop_cols = [
            "LoanNr_ChkDgt",
            "Name",
            "City",
            "Zip",
            "Bank",
            "ApprovalFY",
            "CreateJob",
            "RetainedJob",
            "FranchiseCode",
            "ChgOffDate",
            "DisbursementDate",
            "BalanceGross",
            "ChgOffPrinGr",
        ]
        self.currency_cols = ["DisbursementGross", "GrAppv", "SBA_Appv"]
        self.naics_sector = {
            "11": "Agriculture, forestry, fishing and hunting",
            "21": "Mining, quarrying, and oil and gas extraction",
            "22": "Utilities",
            "23": "Construction",
            "31": "Manufacturing",
            "32": "Manufacturing",
            "33": "Manufacturing",
            "42": "Wholesale trade",
            "44": "Retail trade",
            "45": "Retail trade",
            "48": "Transportation and warehousing",
            "49": "Transportation and warehousing",
            "51": "Information",
            "52": "Finance and insurance",
            "53": "Real estate and rental and leasing",
            "54": "Professional, scientific, and technical services",
            "55": "Management of companies and enterprises",
            "56": "Administrative and support and waste management and remediation services",
            "61": "Educational services",
            "62": "Health care and social assistance",
            "71": "Arts, entertainment, and recreation",
            "72": "Accommodation and food services",
            "81": "Other services (except public administration)",
            "92": "Public administration",
            "0": "[Unallocated sector]",
        }
        self.target_col = "MIS_Status"
        self.new_cols = ["DifState", "Secured", "SecuredSBA"]
        self.rename_cols = {
            "NAICS": "Sector",
            "RevLineCr": "RevLine",
            "DisbursementGross": "GrDisburs",
            "GrAppv": "GrApprov",
            "SBA_Appv": "ApprovSBA",
        }

        self.drop_nans = ["Default"]
        self.mode_cols = ["State", "BankState"]
        self.class_cols = ["NewExist", "RevLine", "LowDoc"]
        self.pred_cols = [
            "_",
            "AppYear",
            "AppMonth",
            "Term",
            "NoEmp",
            "UrbanRural",
            "GrDisburs",
            "GrApprov",
            "ApprovSBA",
        ]

        self.cat_nom_cols = ["State", "BankState", "Sector"]
        self.cat_num_cols = "UrbanRural"
        self.cat_ord_cols = "AppYear"
        self.sort_cols = [
            "State",
            "BankState",
            "DifState",
            "Sector",
            "AppYear",
            "AppMonth",
            "Term",
            "NoEmp",
            "Secured",
            "NewExist",
            "Urban",
            "Rural",
            "RevLine",
            "LowDoc",
            "GrDisburs",
            "GrApprov",
            "ApprovSBA",
            "SecuredSBA",
            "Default",
        ]

        self.num_dis = ["Term", "NoEmp", "SecuredSBA"]
        self.num_con = ["GrDisburs", "GrApprov", "ApprovSBA"]
        self.cat_nom = ["State", "BankState", "Sector"]
        self.cat_num = []
        self.cat_ord = ["AppYear", "AppMonth"]
        self.binary = [
            "DifState",
            "Secured",
            "NewExist",
            "Urban",
            "Rural",
            "RevLine",
            "LowDoc",
        ]
        self.target = ["Default"]

        self.random_sample = self.config["vars"]["rand_sample"]

    def __FeatureTransform(self):
        """Pipeline to transform features"""
        feature_transform = Pipeline(
            [
                ("drop_columns", dt_trn.DropColumns(self.drop_cols)),
                ("currency_int", dt_trn.CurrencyToInt(self.currency_cols)),
                ("approval_date", dt_trn.ApprovalDate("ApprovalDate")),
                (
                    "categorize_naics",
                    dt_trn.CategorizeNAICS("NAICS", self.naics_sector),
                ),
                ("convert_newexist", dt_trn.ConvertNewExist("NewExist")),
                ("convert_revlinecr", dt_trn.ConvertRevLineCr("RevLineCr")),
                ("convert_lowdoc", dt_trn.ConvertLowDoc("LowDoc")),
                ("convert_target", dt_trn.ConvertTarget(self.target_col)),
                ("create_features", dt_trn.CreateFeatures(self.new_cols)),
                ("rename_columns", dt_trn.RenameColumns(self.rename_cols)),
            ]
        )
        return feature_transform

    def __MissingValues(self):
        """Pipeline to impute missing values"""
        missing_values = Pipeline(
            [
                ("mode_imputer", dt_trn.ModeImputer(self.mode_cols)),
                ("class_imputer", dt_trn.ClassImputer(self.class_cols, self.pred_cols)),
                ("drop_duplicates", dt_trn.DropDuplicates(self.target)),
            ]
        )
        return missing_values

    def __EncodeFeatures(self):
        """Pipeline to encode categorical features"""
        encode_features = Pipeline(
            [
                ("label_encoder", dt_trn.LabelTransformer(self.cat_nom_cols)),
                ("one-hot_encoder", dt_trn.OneHotTransformer(self.cat_num_cols)),
                ("ordinal_encoder", dt_trn.OrdinalTransformer(self.cat_ord_cols)),
                ("drop_nans", dt_trn.DropNaNs(self.drop_nans)),
                ("sort_columns", dt_trn.SortColumns(self.sort_cols)),
            ]
        )
        return encode_features

    def PreprocessingPipeline(self):
        """Pipeline to preprocessing dataset"""
        df = pd.read_csv(self.path + self.config["data"]["data_raw"], low_memory=False)

        preprocessing = Pipeline(
            [
                ("feature_transform", self.__FeatureTransform()),
                ("missing_values", self.__MissingValues()),
                ("encode_features", self.__EncodeFeatures()),
            ]
        )

        preprocessing_fit = preprocessing.fit(df)
        return preprocessing_fit

    def UnderSamplingPipeline(self):
        """Pipeline to undersampling dataset"""
        df = pd.read_csv(
            self.path + self.config["data"]["data_train"], low_memory=False
        )

        under_sampling = Pipeline(
            [
                ("feature_transform", self.__FeatureTransform()),
                ("missing_values", self.__MissingValues()),
                ("encode_features", self.__EncodeFeatures()),
                (
                    "under_sampler",
                    dt_trn.SubSampling(self.target[0], self.random_sample),
                ),
            ]
        )

        self.under_sampling_fit = under_sampling.fit(df)
        return self.under_sampling_fit.transform(df)

    def SmoteSamplingPipeline(self):
        """Pipeline to oversampling dataset with SMOTE"""
        df = pd.read_csv(
            self.path + self.config["data"]["data_train"], low_memory=False
        )

        smote_sampling = Pipeline(
            [
                ("feature_transform", self.__FeatureTransform()),
                ("missing_values", self.__MissingValues()),
                ("encode_features", self.__EncodeFeatures()),
                (
                    "smote_sampler",
                    dt_trn.SmoteSampling(self.target[0], self.random_sample),
                ),
            ]
        )

        self.smote_sampling_fit = smote_sampling.fit(df)
        return self.smote_sampling_fit.transform(df)


if __name__ == "__main__":
    try:
        df = pd.read_csv(os.getcwd() + "/data/raw/sba_test.csv", low_memory=False)
        df = df.drop(columns=["MIS_Status"])

        data_pipe = DataPipelines()
        transformer = data_pipe.PreprocessingPipeline()
        df_ = transformer.transform(df)

        print(df_.shape)
        print(df_.sample(3))

    except Exception as err:
        print("Error: ", str(err))
