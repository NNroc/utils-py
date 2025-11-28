from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score


def train_classification(X_train, y_train, X_test, y_test, model_name="svm"):

    models = {
        "svm": SVC(probability=True),
        "dt": DecisionTreeClassifier(),
        "rf": RandomForestClassifier(n_estimators=300),
        "gb": GradientBoostingClassifier()
    }

    model = models[model_name]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "F1_macro": f1_score(y_test, pred, average="macro"),
        "Confusion": confusion_matrix(y_test, pred)
    }

    if proba is not None and len(set(y_test)) == 2:
        metrics["AUC"] = roc_auc_score(y_test, proba[:, 1])

    return model, metrics
