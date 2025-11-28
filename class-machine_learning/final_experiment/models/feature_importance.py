import numpy as np
import pandas as pd


def extract_feature_importance(model, feature_names, top_k=10):
    """
    支持 RF / XGB / 线性模型
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return None

    df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    return df.sort_values("Importance", ascending=False).head(top_k)
