import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# 加载数据集
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# 训练 XGBoost 模型
model = xgb.XGBRegressor()
model.fit(X, y)

# 创建 SHAP 解释器
explainer = shap.Explainer(model)
shap_values = explainer(X)

# 1. 特征重要性图（summary_plot）
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.close()

# 2. 单个样本解释（force_plot）
shap.initjs()
sample_idx = 0
shap.force_plot(explainer.expected_value, shap_values[sample_idx].values, X.iloc[sample_idx], matplotlib=True, show=False)
plt.tight_layout()
plt.savefig("shap_force_plot.png", bbox_inches='tight')
plt.close()
