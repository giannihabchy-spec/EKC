from ml.forecasting.rf import fit_rf
from ml.forecasting.xgboost import fit_xgb
from ml.forecasting.sarima import fit_sarima
from ml.forecasting.nhits import fit_nhits, fit_single_nhits

MODEL_REGISTRY = {
    # "rf":  fit_rf,
    # "xgb": fit_xgb,
    # "sarima": fit_sarima,
    # "nhits": fit_nhits,
}

MULTI_REGISTRY = {
    "nhits_multi": fit_single_nhits,
}
