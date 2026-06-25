from ml.forecasting.rf import fit_rf
from ml.forecasting.xgboost import fit_xgb
from ml.forecasting.sarima import fit_sarima

MODEL_REGISTRY = {
    # "rf":  fit_rf,
    "xgb": fit_xgb,
    # "sarima": fit_sarima,
}