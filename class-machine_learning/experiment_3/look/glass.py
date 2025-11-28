import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

# 加载glass数据集
# 假设glass.data是以逗号分隔的CSV文件，且最后一列为标签
data = pd.read_csv('glass.data', header=None)  # 根据实际情况设置header
X = data.iloc[:, :-1].values  # 提取特征（所有列，除了最后一列）
true_labels = data.iloc[:, -1].values  # 提取真实标签（最后一列）

# 标准化数据
X_scaled = StandardScaler().fit_transform(X)

# 定义一个字典，用于存储不同的聚类方法及其对应的初始化参数和默认参数
clustering_methods = {
    "KMeans": (KMeans(n_clusters=3, random_state=42), {}),  # KMeans聚类，设置3个簇，随机种子为42
    "DBSCAN": (DBSCAN(eps=0.5, min_samples=5), {}),         # DBSCAN聚类，邻域半径为0.5，最小样本数为5
    "Agglomerative": (AgglomerativeClustering(n_clusters=3), {}), # 层次聚类，设置3个簇
    "AP聚类": (AffinityPropagation(damping=0.9), {})        # AP聚类，阻尼系数为0.9
}

# 评估聚类结果
def evaluate_clustering(labels):
    ari = adjusted_rand_score(true_labels, labels)
    silhouette_avg = silhouette_score(X_scaled, labels)
    return ari, silhouette_avg

for name, (method, params) in clustering_methods.items():
    labels = method.fit_predict(X_scaled)

    # 处理DBSCAN和AP聚类的噪声标签
    if name in ["DBSCAN", "AP聚类"]:
        labels = [label if label != -1 else -2 for label in labels]

    # 计算并输出评估指标
    ari, silhouette = evaluate_clustering(labels)
    print(f"{name} - ARI: {ari:.2f}, Silhouette Coefficient: {silhouette:.2f}")
