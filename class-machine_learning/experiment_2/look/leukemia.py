import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# 读取特征变量文件（假设是 CSV 格式）
data = pd.read_csv('../data/LeukemiaDataSet3.dat',sep='\s+',header=None)# features 的形状应该是 (72, 7130)

# 仅使用第二列作为目标变量
feature_data = data.iloc[:, 1:]
target_data = data.iloc[:, 0]  # 第1列作为目标变量

# 确保特征和目标的形状是匹配的
print(f'Features shape: {feature_data.shape}')  # 应为 (72, 7129)
print(f'Target shape: {target_data.shape}')  # 应为 (72,)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.2, random_state=12)


# 定义基本分类器
clf1 = DecisionTreeClassifier(random_state=1)#决策树
clf2 = RandomForestClassifier(random_state=1)#随机森林
clf3 = SVC(probability=True, random_state=1)#SVM
clf4 = LogisticRegression(max_iter=10000,random_state=1)#逻辑回归

# 定义集成方法
bagging = BaggingClassifier(n_estimators=100, random_state=1)#bagging
boosting = GradientBoostingClassifier(random_state=1)#Gradient Boosting

# 训练并评估
classifiers = [clf1, clf2, clf3, clf4, bagging, boosting]
classifier_names = ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression', 'bagging','Gradient Boosting']
results = {}

# 训练模型并预测
for clf, name in zip(classifiers, classifier_names):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.4f}')

# 计算每个分类器的交叉验证精度
weights = []
classifiers1 = [clf1, clf2, clf3, clf4]
classifiers2 = [clf1, clf2, clf3]
classifiers3 = [clf1, clf2, clf4]
classifiers4 = [clf2, clf3]
classifiers5 = [clf1, clf4]
classes = [classifiers1, classifiers2, classifiers3, classifiers4, classifiers5]
# 定义分类器名称
classifier_names = ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression']

# 用于存储每个分类器组的归一化权重
normalized_weights_list = []

# 分别计算每个分类器组的权重
for i in range(len(classes)):
    weights = []  # 每组的权重
    print(f'Weights for Classifier Group {i + 1}:')

    for clf in classes[i]:
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)  # 5折交叉验证
        weights.append(cv_scores.mean())  # 使用平均精度作为权重

    # 归一化权重
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]

    # 将归一化权重存储到总数组中
    normalized_weights_list.append(normalized_weights)

    # 输出每个分类器的权重
    for name, weight, normalized_weight in zip(classifier_names, weights, normalized_weights):
        print(f'{name} Cross-Validation Accuracy: {weight:.4f}, Normalized Weight: {normalized_weight:.4f}')

    print()  # 空行分隔不同组的输出

# 打印归一化权重列表
print("Normalized Weights for Each Classifier Group:")
for i, normalized_weights in enumerate(normalized_weights_list):
    print(f'Group {i + 1}: {normalized_weights}')

# Soft和Hard集成分类器1234
voting_soft = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Random Forest', clf2),
    ('SVM', clf3),
    ('Logistic Regression', clf4),
], voting='soft',weights=normalized_weights_list[0])

voting_soft00 = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Random Forest', clf2),
    ('SVM', clf3),
    ('Logistic Regression', clf4),
], voting='soft')

voting_hard = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Random Forest', clf2),
    ('SVM', clf3),
    ('Logistic Regression', clf4),
], voting='hard')

# Soft和Hard集成分类器123
voting_soft1 = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Random Forest', clf2),
    ('SVM', clf3),
], voting='soft',weights=normalized_weights_list[1])

voting_soft01 = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Random Forest', clf2),
    ('SVM', clf3),
], voting='soft')

voting_hard1 = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Random Forest', clf2),
    ('SVM', clf3),
], voting='hard')

# Soft和Hard集成分类器124
voting_soft2 = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Random Forest', clf2),
    ('Logistic Regression', clf4),
], voting='soft',weights=normalized_weights_list[2])

voting_soft02 = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Random Forest', clf2),
    ('Logistic Regression', clf4),
], voting='soft')

voting_hard2 = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Random Forest', clf2),
    ('Logistic Regression', clf4),
], voting='hard')

# Soft和Hard集成分类器14
voting_soft3 = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Logistic Regression', clf4),
], voting='soft',weights=normalized_weights_list[3])

voting_soft03 = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Logistic Regression', clf4),
], voting='soft')

voting_hard3 = VotingClassifier(estimators=[
    ('Decision Tree', clf1),
    ('Logistic Regression', clf4),
], voting='hard')

# Soft和Hard集成分类器23
voting_soft4 = VotingClassifier(estimators=[
    ('Random Forest', clf2),
    ('SVM', clf3),
], voting='soft',weights=normalized_weights_list[4])

voting_soft04 = VotingClassifier(estimators=[
    ('Random Forest', clf2),
    ('SVM', clf3),
], voting='soft')

voting_hard4 = VotingClassifier(estimators=[
    ('Random Forest', clf2),
    ('SVM', clf3),
], voting='hard')

# 训练模型并评估
classifiers6 = {
    "加权后soft集成1234": voting_soft,
    "soft集成1234": voting_soft00,
    "hard集成1234": voting_hard,
    "加权后soft集成123": voting_soft1,
    "soft集成123": voting_soft01,
    "hard集成123": voting_hard1,
    "加权后soft集成124": voting_soft2,
    "soft集成124": voting_soft02,
    "hard集成124": voting_hard2,
    "加权后soft集成14": voting_soft3,
    "soft集成14": voting_soft03,
    "hard集成14": voting_hard3,
    "加权后soft集成23": voting_soft4,
    "soft集成23": voting_soft04,
    "hard集成23": voting_hard4,
}

# 训练模型并预测
for name, clf in classifiers6.items():
    clf.fit(X_train, y_train)  # 训练模型
    y_pred = clf.predict(X_test)  # 进行预测
    accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
    print(f'{name} Accuracy: {accuracy:.4f}')  # 输出结果
