config_map = { # preprocesor name : sheet name in supa/config
    'sales by items': 'Sales',
    'sales by category': 'Sales. Cat.',
    'monthly sales by items': 'Sales',
    'daily sales by group': 'Daily Sales'
}

sheet_config = {
       "results": { ############################################################################################
        "target_table": "forecast_daily_sales_results",

        "expected_columns": [
            'branch_id',
            'category',
            'freq',
            'model',
            'report_date',
            'val_wape',
            'final_mae',
            'final_rmse',
            'final_wape',
            'is_best',
            'result'
        ],

        "unique_key": [
            "branch_id",
            "report_date",
            "category",
            'model',
            'freq'
        ],

        "load_mode": "insert",

        "no_nulls": [
            'category', 
            'sales'
        ]
    },
}


SEASONAL_PERIOD = {"D": 7, "W": 52}
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15


D_LAGS      = [1, 2, 3, 7, 14, 28, 90, 365]
W_LAGS      = [1, 2, 3, 4, 8, 12, 13, 26, 52]
D_ROLLS     = [7, 14, 30, 90]
W_ROLLS     = [4, 8, 13, 26, 52]


rf_tuning = {
    'max_depth': [5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ["sqrt", "log2", 0.5, 0.8],
    'n_est': [100, 300],
}

xgb_tuning = {
    'n_est': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

nhits_tuning = {
    "lookback": [30, 60],
    "hidden_size": [64, 128],
    "n_stacks": [2, 3],
    "lr": [0.001, 0.005],
}

tft_tuning = {
    "lookback": [30, 60],
    "hidden_size": [64, 128],
    "n_heads": [2, 4],
    "lr": [0.001, 0.005],
}




# D_LAGS      = [1, 2, 3, 7, 14]
# W_LAGS      = [1, 2, 3, 4]
# D_ROLLS     = [7, 14, 30]
# W_ROLLS     = [4, 8]


# rf_tuning = {
#     'max_depth': [5],
#     'min_samples_leaf': [1],
#     'max_features': ["sqrt"],
#     'n_est': [10],
# }

# xgb_tuning = {
#     'n_est': [10],
#     'learning_rate': [0.2],
#     'max_depth': [3],
#     'subsample': [0.8],
#     'colsample_bytree': [0.8],
# }

# nhits_tuning = {
#     "lookback": [30],
#     "hidden_size": [64],
#     "n_stacks": [2],
#     "lr": [0.001],
# }

# tft_tuning = {
#     "lookback": [30],
#     "hidden_size": [64],
#     "n_heads": [2],
#     "lr": [0.005],
# }