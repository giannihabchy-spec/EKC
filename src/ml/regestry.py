from ml.forecasting.rf import fit_rf
from ml.forecasting.xgboost import fit_xgb
from ml.forecasting.sarima import fit_sarima
from ml.forecasting.nhits import fit_nhits, fit_single_nhits
from ml.forecasting.tft import fit_tft, fit_single_tft
from ml.forecasting.lgbm import fit_lgbm
from ml.forecasting.xgb_multi import fit_single_xgb

MODEL_REGISTRY = {
    # "sarima": fit_sarima,
    "rf":  fit_rf,
    "xgb": fit_xgb,
    "nhits": fit_nhits,
    "tft": fit_tft,
    "lgbm": fit_lgbm,
}

MULTI_REGISTRY = {
    # "nhits_multi": fit_single_nhits,
    # "tft_multi": fit_single_tft,
    # "xgb_multi": fit_single_xgb,
}
