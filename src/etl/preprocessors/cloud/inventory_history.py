# Remove grouping by details
# nhar l jarde

import pandas as pd
from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import drop_na_by_name
from etl.utils import remove_repeated_headers
from etl.utils import make_columns_numeric
from etl.utils import get_omega_client_name, get_file_date


def preprocess(path, omega_loc: bool = False):
    data = read(path)

    if omega_loc:
        omega_client = get_omega_client_name(data)
        file_date = pd.to_datetime(data.iloc[2,1].split()[-1].replace(':',''))

    data = keep_cols_by_index(data,[1,2,9])
    data.columns = ['product description','qty','location']
    data = drop_na_by_name(data,['qty'])
    data = remove_repeated_headers(data,'qty')
    data = make_columns_numeric(data,['qty'])

    if omega_loc:
        data['omega name'] = omega_client
        data['file date'] = file_date

    return data