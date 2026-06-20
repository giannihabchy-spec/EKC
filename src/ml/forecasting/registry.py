from ml.forecasting.rf import fit_rf
from ml.forecasting.xgboost import fit_xgb

# Add or remove models here — each value must be a callable fit_*(s: pd.Series) -> dict
MODEL_REGISTRY = {
    "rf":  fit_rf,
    "xgb": fit_xgb,
}
