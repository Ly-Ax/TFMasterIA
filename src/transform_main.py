"""Example module"""
import pandas as pd
import os
import yaml
from transform import dataset_load


class DataLoad:
    """Example class"""

    def __init__(self, config):
        """Example function"""
        self.path = os.getcwd()

        with open(config, "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

    def get_df(self):
        """Example function"""
        df = pd.read_csv(self.path + self.config["data"]["data_raw"], low_memory=False)
        return df
    
    def HoldOutMethod(self):
        # df_train, df_temp = train_test_split(df, train_size=0.7, random_state=random_seed)
        # df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=random_seed)

        # df_train.to_csv(data_train, index=False)
        # df_val.to_csv(data_val, index=False)
        # df_test.to_csv(data_test, index=False)

        # print(f"Train: {df_train.shape}")
        # print(f"Val:   {df_val.shape}")
        # print(f"Test:  {df_test.shape}")
        pass

    def SavePipeline(self):
        # joblib.dump(preprocessing_fit, prepro_pipe)

        # prepro_pipeline = joblib.load(prepro_pipe)

        # try:
        #     prepro_pipeline

        #     dt_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #     print(f"{dt_now}: Pipeline is loaded...")

        # except Exception as e:
        #     print("Error:", str(e))
        pass

    def TestPipeline(self):
        # sba_raw = pd.read_csv(data_raw, low_memory=False)
        # sba_clean = prepro_pipeline.transform(sba_raw)

        # sba_clean.to_csv(data_clean, index=False)
        # print(sba_clean.info())
        # sba_clean.sample(3)

        # sba_train = pd.read_csv(data_train, low_memory=False)
        # sba_clean_train = prepro_pipeline.transform(sba_train)

        # sba_clean_train.to_csv(clean_train, index=False)
        # print(sba_clean_train.info())
        # sba_clean_train.sample(3)

        # df_under.to_csv(train_subsam, index=False)
        # print(df_under.shape)
        # print(df_under["Default"].value_counts())
        # df_under.sample(3)

        # df_smote.to_csv(train_smote, index=False)
        # print(df_smote.shape)
        # print(df_smote["Default"].value_counts())
        # df_smote.sample(3)
        pass


if __name__ == "__main__":
    cnf_data = DataLoad("config.yaml")
    df = cnf_data.get_df()
    print(df.sample(3))
