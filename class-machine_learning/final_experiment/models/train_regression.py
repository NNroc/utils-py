from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


def train_regression(X_train, y_train, X_test, y_test, model_name="ridge"):
    models = {
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01),
        "rf": RandomForestRegressor(n_estimators=300, random_state=42),
        "xgb": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
    }

    model = models[model_name]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, pred, squared=False)
    r2 = r2_score(y_test, pred)
    corr = pearsonr(pred, y_test)[0]

    return model, {"RMSE": rmse, "R2": r2, "Pearson": corr}
