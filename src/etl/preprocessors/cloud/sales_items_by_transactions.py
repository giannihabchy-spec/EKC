import pandas as pd
from etl.utils import (
    read,
    keep_cols_by_index,
    drop_rows,
    drop_na_by_name,
    make_columns_numeric,
    make_columns_date,
    clean_check
)


def preprocess(path):
    data = read(path)
    data = keep_cols_by_index(data, [0,1,2,3,4,6,7,10])
    data.columns = ['invoice', 'table', 'employee', 'date_time', 'qty', 'amount', 'description', 'remark']
    data = drop_rows(data,'qty','Qty')
    data = drop_na_by_name(data, ['qty'])
    data['date_time'] = pd.to_datetime(data['date_time'])
    return data