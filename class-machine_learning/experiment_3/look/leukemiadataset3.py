import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

# 读取特征变量文件（假设是 CSV 格式）
data = pd.read_csv('LeukemiaDataSet3.dat', sep='\s+', header=None)
true_labels = data.iloc[:, 0].values  # 提取真实标签（第一列）
X = data.iloc[:, 1:].values  # 提取特征（所有列，除了第一列）

# 标准化数据
X_scaled = StandardScaler().fit_transform(X)

# 定义一个字典，用于存储不同的聚类方法及其对应的初始化参数和默认参数
clustering_methods = {
    "KMeans": (KMeans(n_clusters=3, random_state=42), {}),  # KMeans聚类，设置3个簇，随机种子为42
    "DBSCAN": (DBSCAN(eps=0.5, min_samples=5), {}),         # DBSCAN聚类，邻域半径为0.5，最小样本数为5
    "层次聚类": (AgglomerativeClustering(n_clusters=3), {}), # 层次聚类，设置3个簇
    "AP聚类": (AffinityPropagation(damping=0.9), {})        # AP聚类，阻尼系数为0.9
}

# 评估聚类结果
def evaluate_clustering(labels):
    ari = adjusted_rand_score(true_labels, labels)
    silhouette_avg = silhouette_score(X_scaled, labels)
    return ari, silhouette_avg

# 遍历每种聚类方法并进行评估
for name, (method, params) in clustering_methods.items():
    labels = method.fit_predict(X_scaled)

    # 处理DBSCAN和AP聚类的噪声标签
    if name in ["DBSCAN", "Affinity Propagation"]:
        labels = [label if label != -1 else -2 for label in labels]

    # 检查标签的数量
    unique_labels = set(labels)
    if len(unique_labels) <= 1:
        print(f"{name} encountered an error: Number of unique labels is {len(unique_labels)}. Skipping evaluation.")
        continue

    # 计算并输出评估指标
    try:
        ari, silhouette = evaluate_clustering(labels)
        print(f"{name} - ARI: {ari:.2f}, Silhouette Coefficient: {silhouette:.2f}")
    except ValueError as e:
        print(f"{name} encountered an error: {e}")
