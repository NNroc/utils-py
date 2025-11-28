import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def eda_overview(df: pd.DataFrame):
    print("\n=== 数据基本信息 ===")
    print(df.info())
    print("\n=== 缺失值统计 ===")
    print(df.isna().mean().sort_values(ascending=False).head(20))

def plot_feature_distribution(df: pd.DataFrame, feature_list):
    """
    绘制部分特征的直方图
    """
    df[feature_list].hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

def plot_correlation(df: pd.DataFrame, target):
    """
    绘制和目标变量的相关性热力图
    """
    plt.figure(figsize=(12, 10))
    corr = df.corr()[target].sort_values(ascending=False)[:40]
    sns.barplot(x=corr.values, y=corr.index)
    plt.title(f"Top correlated features with {target}")
    plt.show()
