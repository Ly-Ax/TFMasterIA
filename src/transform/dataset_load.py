"""Module to load dataset"""
import pandas as pd
import os


class DataLoad:
    """Class data load"""

    def __init__(self):
        """Get parent path"""
        self.path = os.getcwd()

    def data_load(self, data_path):
        """Load dataset"""
        df = pd.read_csv(self.path + data_path, low_memory=False)
        return df


if __name__ == "__main__":
    try:
        data = DataLoad()
        df = data.data_load("/data/raw/sba_national.csv")
        print(f"Dims: {df.shape}")

    except Exception as err:
        print("Error: ", str(err))
