import xai, shap, lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
clf = RandomForestClassifier().fit(X, y)

# 选一条样本
x = X.iloc[0:1]

# LIME 局部解释
lime_exp = xai.lime.explain_instance(x, clf.predict_proba, X.columns)
lime_exp.show_in_notebook()  # Jupyter 内直接出条形图

# SHAP 局部解释
explainer = shap.Explainer(clf)
shap.force_plot(explainer.expected_value[1], explainer(x).values[:, :, 1], x)
