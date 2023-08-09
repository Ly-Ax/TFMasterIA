import pandas as pd
import os
import yaml

class DataLoad():
    def __init__(self, config):
        self.path = os.getcwd()

        with open(config, 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)

    def get_df(self):
        return pd.read_csv(self.path + self.config['data']['data_path'], low_memory=False)

if __name__ == '__main__':
    cnf_data = DataLoad('config.yaml')
    df = cnf_data.get_df()
    print(df.sample(3))
