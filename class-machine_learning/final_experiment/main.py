from models.data_loader import load_dataset
from models.eda import *
from models.preprocess import *
from models.split import *
from models.train_regression import train_regression
from models.train_classification import train_classification
from models.feature_importance import extract_feature_importance

CELL_LINES = ["CO_HT29", "CO_HCT-116"]

for cell in CELL_LINES:

    print(f"\n======================")
    print(f"🔥 处理细胞系：{cell}")
    print(f"======================")

    df = load_dataset(f"./data/{cell}/dataset_features.csv")

    # === 1. EDA ===
    eda_overview(df)
    plot_feature_distribution(df, df.columns[5:15])
    plot_correlation(df, "Synergy_Score")

    # === 2. 预处理 ===
    df = handle_missing_values(df, strategy="knn")
    df = feature_selection(df, threshold=0.0, exclude_cols=["Drug_A", "Drug_B", "Synergy_Score", "Class_Label"])
    df = scale_features(df, exclude_cols=["Drug_A", "Drug_B", "Synergy_Score", "Class_Label"])

    # 准备数据
    X = df.drop(columns=["Synergy_Score", "Class_Label", "Drug_A", "Drug_B"])
    y_reg = df["Synergy_Score"]
    y_clf = df["Class_Label"]

    # === 3. 交叉验证（回归+分类） ===
    splits = cv_split(df, n_splits=5)

    reg_results = []
    clf_results = []

    for tr, te in splits:
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train_reg, y_test_reg = y_reg.iloc[tr], y_reg.iloc[te]
        y_train_clf, y_test_clf = y_clf.iloc[tr], y_clf.iloc[te]

        # 回归模型：XGB
        model_reg, metrics_reg = train_regression(X_train, y_train_reg, X_test, y_test_reg, model_name="xgb")
        reg_results.append(metrics_reg)

        # 分类模型：RF
        model_clf, metrics_clf = train_classification(X_train, y_train_clf, X_test, y_test_clf, model_name="rf")
        clf_results.append(metrics_clf)

    print("\n=== 回归平均性能 ===")
    print(pd.DataFrame(reg_results).mean())

    print("\n=== 分类平均性能 ===")
    print(pd.DataFrame([{k: v for k, v in m.items() if k != 'Confusion'} for m in clf_results]).mean())

    # === 4. 特征重要性 ===
    fi = extract_feature_importance(model_reg, X.columns)
    print("\n=== 回归模型特征重要性 Top 10 ===")
    print(fi)
