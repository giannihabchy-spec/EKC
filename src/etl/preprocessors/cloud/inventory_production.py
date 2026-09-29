import pandas as pd
from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import remove_repeated_headers
from etl.utils import drop_na_by_name
from etl.utils import make_columns_date, make_columns_numeric
from etl.utils import get_from_to_date_c1


def preprocess(path):
    data = read(path)

    get_from_to_date_c1(data, (2,3))

    data = keep_cols_by_index(data,[0,4,7,11])
    data.columns = ['date', 'location', 'qty', 'production list']
    data = remove_repeated_headers(data,'date')
    data = drop_na_by_name(data,['location'])
    data = drop_na_by_name(data,['qty'])
    data = make_columns_date(data,['date'])
    data = make_columns_numeric(data,['qty'])
    return data