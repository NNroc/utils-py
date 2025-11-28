from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data

# 标准化数据
X_scaled = StandardScaler().fit_transform(X)

# 定义一个字典，用于存储不同的聚类方法及其对应的初始化参数和默认参数
clustering_methods = {
    "KMeans": (KMeans(n_clusters=3, random_state=42), {}),  # KMeans聚类，设置3个簇，随机种子为42
    "DBSCAN": (DBSCAN(eps=0.5, min_samples=5), {}),         # DBSCAN聚类，邻域半径为0.5，最小样本数为5
    "层次聚类": (AgglomerativeClustering(n_clusters=3), {}), # 层次聚类，设置3个簇
    "AP聚类": (AffinityPropagation(damping=0.9), {})        # AP聚类，阻尼系数为0.9
}

# 真实标签
true_labels = iris.target


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
