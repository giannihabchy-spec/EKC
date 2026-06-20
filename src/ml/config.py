config_map = { # preprocesor name : sheet name in supa/config
    'sales by items': 'Sales',
    'sales by category': 'Sales. Cat.',
    'monthly sales by items': 'Sales',
    'daily sales by group': 'Daily Sales'
}

SEASONAL_PERIOD = 7
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# LAGS     = [1, 2, 3, 7, 14, 21, 28]
LAGS     = [1,2,3,7,14,21,28,35,42,49,56]
ROLLS    = [7, 14, 28, 56]


sheet_config = {
       "results": { ############################################################################################
        "target_table": "forecast_daily_sales_results",

        "expected_columns": [
            'branch_id',
            'category',
            'model',
            'report_date',
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
            'model'
        ],

        "load_mode": "insert",

        "no_nulls": [
            'category', 
            'sales'
        ]
    },
}