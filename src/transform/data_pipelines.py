"""Main module from data pipelines"""
import sys
from pathlib import Path
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).parents[1]))

from transform import dataset_load as dt_ld
from transform import feature_transform as ft_tr
from transform import missing_values as ms_vl
from transform import encode_features as en_ft
from transform import resampling_train as rs_tr


class DataPipelines:
    """Pipelines to transform datasets"""

    def __init__(self):
        """Initialize variables"""
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

    def __DataRetrieval(self, data_path):
        """Load the specified dataset"""
        dt_load = dt_ld.DataLoad()
        self.df = dt_load.data_load(data_path)

    def __FeatureTransform(self):
        """Pipeline to transform features"""
        feature_transform = Pipeline(
            [
                ("drop_columns", ft_tr.DropColumns(self.drop_cols)),
                ("currency_int", ft_tr.CurrencyToInt(self.currency_cols)),
                ("approval_date", ft_tr.ApprovalDate("ApprovalDate")),
                ("categorize_naics", ft_tr.CategorizeNAICS("NAICS", self.naics_sector)),
                ("convert_newexist", ft_tr.ConvertNewExist("NewExist")),
                ("convert_revlinecr", ft_tr.ConvertRevLineCr("RevLineCr")),
                ("convert_lowdoc", ft_tr.ConvertLowDoc("LowDoc")),
                ("convert_target", ft_tr.ConvertTarget(self.target_col)),
                ("create_features", ft_tr.CreateFeatures(self.new_cols)),
                ("rename_columns", ft_tr.RenameColumns(self.rename_cols)),
            ]
        )
        return feature_transform

    def __MissingValues(self):
        """Pipeline to impute missing values"""
        missing_values = Pipeline(
            [
                ("mode_imputer", ms_vl.ModeImputer(self.mode_cols)),
                ("class_imputer", ms_vl.ClassImputer(self.class_cols, self.pred_cols)),
                ("drop_duplicates", ms_vl.DropDuplicates()),
            ]
        )
        return missing_values

    def __EncodeFeatures(self):
        """Pipeline to encode categorical features"""
        encode_features = Pipeline(
            [
                ("label_encoder", en_ft.LabelTransformer(self.cat_nom_cols)),
                ("one-hot_encoder", en_ft.OneHotTransformer(self.cat_num_cols)),
                ("ordinal_encoder", en_ft.OrdinalTransformer(self.cat_ord_cols)),
                ("drop_nans", en_ft.DropNaNs(self.drop_nans)),
                ("sort_columns", en_ft.SortColumns(self.sort_cols)),
            ]
        )
        return encode_features

    def PreprocessingPipeline(self, data_path):
        """Pipeline to preprocessing dataset"""
        self.__DataRetrieval(data_path)

        preprocessing = Pipeline(
            [
                ("feature_transform", self.__FeatureTransform()),
                ("missing_values", self.__MissingValues()),
                ("encode_features", self.__EncodeFeatures()),
            ]
        )

        self.preprocessing_fit = preprocessing.fit(self.df)
        return self.preprocessing_fit.transform(self.df)

    def PreprocessingPipelineFit(self, data_path):
        """Pipeline to preprocessing dataset"""
        self.__DataRetrieval(data_path)

        preprocessing = Pipeline(
            [
                ("feature_transform", self.__FeatureTransform()),
                ("missing_values", self.__MissingValues()),
                ("encode_features", self.__EncodeFeatures()),
            ]
        )

        return preprocessing.fit(self.df)

    def UnderSamplingPipeline(self, data_path, random_sample):
        """Pipeline to undersampling dataset"""
        self.__DataRetrieval(data_path)

        under_sampling = Pipeline(
            [
                ("feature_transform", self.__FeatureTransform()),
                ("missing_values", self.__MissingValues()),
                ("encode_features", self.__EncodeFeatures()),
                ("under_sampler", rs_tr.SubSampling(self.target[0], random_sample)),
            ]
        )

        self.under_sampling_fit = under_sampling.fit(self.df)
        return self.under_sampling_fit.transform(self.df)

    def SmoteSamplingPipeline(self, data_path, random_sample):
        """Pipeline to oversampling dataset with SMOTE"""
        self.__DataRetrieval(data_path)

        smote_sampling = Pipeline(
            [
                ("feature_transform", self.__FeatureTransform()),
                ("missing_values", self.__MissingValues()),
                ("encode_features", self.__EncodeFeatures()),
                (
                    "smote_sampler",
                    rs_tr.SmoteSampling(self.target[0], random_sample),
                ),
            ]
        )

        self.smote_sampling_fit = smote_sampling.fit(self.df)
        return self.smote_sampling_fit.transform(self.df)


if __name__ == "__main__":
    try:
        data_pipe = DataPipelines()
        df = data_pipe.PreprocessingPipeline("/data/raw/sba_national.csv")

        print(df.shape)
        print(df.sample(3))

    except Exception as err:
        print("Error: ", str(err))
