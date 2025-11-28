import numpy as np
from sklearn.model_selection import KFold


def cv_split(df, n_splits=5):
    """
    常规 K 折交叉验证
    返回：索引列表
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(kf.split(df))


def leave_drugs_out_split(df, test_drugs):
    """
    按药物留一法：所有涉及 test_drugs 的组合进入测试集
    """
    mask = df["Drug_A"].isin(test_drugs) | df["Drug_B"].isin(test_drugs)
    test_df = df[mask]
    train_df = df[~mask]
    return train_df, test_df
