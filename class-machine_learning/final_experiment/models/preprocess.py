import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


def handle_missing_values(df: pd.DataFrame, strategy="knn"):
    """
    缺失值处理：支持 knn / mean
    """
    if strategy == "mean":
        return df.fillna(df.mean())

    imputer = KNNImputer(n_neighbors=5)
    df[df.columns] = imputer.fit_transform(df)
    return df


def scale_features(df: pd.DataFrame, exclude_cols):
    """
    标准化特征（除标签列和药物名）
    """
    scaler = StandardScaler()

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df


def feature_selection(df: pd.DataFrame, threshold=0.0, exclude_cols=[]):
    """
    方差过滤，剔除无信息特征
    """
    selector = VarianceThreshold(threshold)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    selected = selector.fit_transform(df[feature_cols])
    new_cols = [c for c, keep in zip(feature_cols, selector.get_support()) if keep]

    df = df[exclude_cols + new_cols]
    return df
