"""Dataset transformation"""
import pandas as pd
import os
import yaml
import joblib
from transform import data_pipelines
from sklearn.model_selection import train_test_split


class TransformData:
    """Class to apply the transformations"""

    def __init__(self):
        """Initialize variables"""
        self.path = os.getcwd()
        with open("config.yaml", "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

    def HoldOutMethod(self):
        """Split data with Hold-Out method"""
        df = pd.read_csv(self.path + self.config["data"]["data_raw"], low_memory=False)

        df_train, df_temp = train_test_split(
            df,
            train_size=self.config["vars"]["train_split"],
            random_state=self.config["vars"]["rand_holdout"],
        )

        df_val, df_test = train_test_split(
            df_temp,
            test_size=self.config["vars"]["test_split"],
            random_state=self.config["vars"]["rand_holdout"],
        )

        df_train.to_csv(self.path + self.config["data"]["data_train"], index=False)
        df_val.to_csv(self.path + self.config["data"]["data_val"], index=False)
        df_test.to_csv(self.path + self.config["data"]["data_test"], index=False)

        print(f"Train: {df_train.shape}")
        print(f"Val:   {df_val.shape}")
        print(f"Test:  {df_test.shape}")

    def SavePipeline(self):
        """Save preprocessing main pipeline"""
        try:
            dt_trn = data_pipelines.DataPipelines()
            prepro_fit = dt_trn.PreprocessingPipeline()

            joblib.dump(prepro_fit, self.path + self.config["models"]["preprocessing"])

            prepro_pipe = joblib.load(
                self.path + self.config["models"]["preprocessing"]
            )
            prepro_pipe

        except Exception as err:
            print("Error:", str(err))

    def Preprocessing(self, df):
        """Persistent preprocessing main pipeline"""
        prepro_pipe = joblib.load(self.path + self.config["models"]["preprocessing"])

        df_ = prepro_pipe.transform(df)
        return df_

    def GenerateCleanData(self):
        """Generate: sba_national.csv to sba_clean.csv"""
        prepro_pipe = joblib.load(self.path + self.config["models"]["preprocessing"])

        sba_raw = pd.read_csv(
            self.path + self.config["data"]["data_raw"], low_memory=False
        )
        sba_clean = prepro_pipe.transform(sba_raw)
        sba_clean.to_csv(self.path + self.config["data"]["data_clean"], index=False)
        print(f"Data Clean: {sba_clean.shape}")

        """Generate: sba_train.csv to clean_train.csv"""
        sba_train = pd.read_csv(
            self.path + self.config["data"]["data_train"], low_memory=False
        )
        sba_clean_train = prepro_pipe.transform(sba_train)
        sba_clean_train.to_csv(
            self.path + self.config["data"]["clean_train"], index=False
        )
        print(f"Clean Train: {sba_clean_train.shape}")

        """Generate: sba_val.csv to clean_val.csv"""
        sba_val = pd.read_csv(
            self.path + self.config["data"]["data_val"], low_memory=False
        )
        sba_clean_val = prepro_pipe.transform(sba_val)
        sba_clean_val.to_csv(self.path + self.config["data"]["clean_val"], index=False)
        print(f"Clean Val: {sba_clean_val.shape}")

        """Generate: sba_test.csv to clean_test.csv"""
        sba_test = pd.read_csv(
            self.path + self.config["data"]["data_test"], low_memory=False
        )
        sba_clean_test = prepro_pipe.transform(sba_test)
        sba_clean_test.to_csv(
            self.path + self.config["data"]["clean_test"], index=False
        )
        print(f"Clean Test: {sba_clean_test.shape}")

    def GenerateResampling(self):
        """SubSampling: sba_train.csv to train_subsam.csv"""
        data_pipe = data_pipelines.DataPipelines()

        df_under = data_pipe.UnderSamplingPipeline()
        df_under.to_csv(self.path + self.config["data"]["train_subsam"], index=False)
        print(df_under.shape)
        print(df_under["Default"].value_counts())

        """Generate SMOTE: sba_train.csv to train_smote.csv"""
        df_smote = data_pipe.SmoteSamplingPipeline()
        df_smote.to_csv(self.path + self.config["data"]["train_smote"], index=False)
        print(df_smote.shape)
        print(df_smote["Default"].value_counts())


if __name__ == "__main__":
    try:
        data = TransformData()
        # data.HoldOutMethod()
        # data.SavePipeline()
        # data.GenerateCleanData()
        # data.GenerateResampling()
        
        df = pd.read_csv(os.getcwd() + "/data/raw/sba_test.csv", low_memory=False)
        # df = df.drop(columns=["MIS_Status"])
        df = data.Preprocessing(df)

        print(df.shape)
        print(df.sample(3))

    except Exception as err:
        print("Error: ", str(err))
