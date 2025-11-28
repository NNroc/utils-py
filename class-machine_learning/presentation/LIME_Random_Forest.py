import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# 加载葡萄酒质量数据集（白葡萄酒）
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
data = pd.read_csv(url, sep=';')

# 构建二分类任务：好（>=7） vs 差（<7）
data['quality_label'] = (data['quality'] >= 7).astype(int)
X = data.drop(['quality', 'quality_label'], axis=1)
y = data['quality_label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练随机森林模型
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# 创建 LIME 解释器
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns,
    class_names=['Bad', 'Good'],
    mode='classification'
)

# 解释单个样本
idx = 0
exp = explainer.explain_instance(X_test_scaled[idx], model.predict_proba, num_features=10)

# 保存解释图
fig = exp.as_pyplot_figure()
plt.title("LIME Explanation for Wine Quality Prediction")
plt.tight_layout()
plt.savefig("lime_explanation.png")
plt.close()