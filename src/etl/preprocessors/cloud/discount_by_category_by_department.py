import pandas as pd
from etl.utils import read 
from etl.utils import keep_cols_by_index
from etl.utils import drop_na_by_name
from etl.utils import make_columns_numeric
from etl.utils import get_from_to_date_1, validate_from_to_date

def preprocess(path):
    data = read(path)

    from_date, to_date = get_from_to_date_1(data, (2,2))
    validate_from_to_date(from_date, to_date)

    data = data.iloc[:,[1,-3]].copy()
    data.columns = ['category', 'total']
    data = drop_na_by_name(data,['category'])
    data = make_columns_numeric(data,['total'])
    return data