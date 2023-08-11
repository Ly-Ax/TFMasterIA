"""Example module"""
import pandas as pd
import os
import yaml


class DataLoad:
    """Example class"""

    def __init__(self, config):
        """Example function"""
        self.path = os.getcwd()

        with open(config, "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

    def get_df(self):
        """Example function"""
        df = pd.read_csv(self.path + self.config["data"]["data_path"], low_memory=False)
        return df


if __name__ == "__main__":
    cnf_data = DataLoad("config.yaml")
    df = cnf_data.get_df()
    print(df.sample(3))
