import pandas as pd

def load_dataset(features_path: str):
    """
    加载特征宽表（dataset_features.csv）
    返回：DataFrame
    """
    df = pd.read_csv(features_path)
    return df
