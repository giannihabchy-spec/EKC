import pandas as pd
from etl.utils import read
from etl.utils import drop_na_by_name
from etl.utils import make_columns_numeric
from etl.utils import make_columns_date
from etl.utils import get_file_date
from etl.utils import get_omega_client_name

def preprocess(path):
    data = read(path)
    dt = get_file_date(data, (9,7), 'local')
    omega_name = get_omega_client_name(data, (1,0))
    col_idx = 2
    first_row = data.iloc[:, col_idx].first_valid_index()
    last_row = data.iloc[:, col_idx].last_valid_index()
    row_idx = 12
    valid_cols = data.iloc[row_idx].notna()
    first_col = valid_cols.idxmax()
    last_col = valid_cols[::-1].idxmax()
    data = data.iloc[first_row : last_row + 1, first_col : last_col + 1].copy()
    data = data.iloc[:,[0,-1]].copy()
    data.columns = ['category', 'sales']
    data['sales'] = data['sales'].shift(-1)
    data = drop_na_by_name(data, ['category'])
    data = make_columns_numeric(data, ['sales'])
    sum_row = pd.DataFrame({
        'category': ['calculated_sum'],
        'sales': [data.sales.sum()]
    })
    data = pd.concat([data, sum_row], ignore_index=True)
    data['omega_name'] = omega_name
    data['report_date'] = dt
    data = make_columns_date(data, ['report_date'])
    cols = ['omega_name', 'report_date', 'category', 'sales']
    data = data[cols]
    return data