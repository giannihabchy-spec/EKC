import pandas as pd
from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import drop_na_by_name
from etl.utils import drop_rows
from etl.utils import make_columns_numeric
from etl.utils import get_omega_client_name
from etl.utils import get_file_date
from etl.utils import make_columns_date


def preprocess(path):
    data = read(path)
    dt = get_file_date(data, (2,2))
    omega_name = get_omega_client_name(data)
    total_col = int(data.iloc[3].last_valid_index())
    data = keep_cols_by_index(data,[0,total_col])
    data.columns = ['category', 'sales']
    data = drop_na_by_name(data,['category','sales'])
    data = drop_rows(data,'category',value = 'Total ')
    data = make_columns_numeric(data,['sales'])
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