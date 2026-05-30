from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import drop_na_by_name
from etl.utils import make_columns_numeric
from etl.utils import get_file_date, get_omega_client_name


def preprocess(path, omega_loc: bool = False):
    data = read(path)

    if omega_loc:
        omega_client = get_omega_client_name(data, (1,0))
        file_date = get_file_date(data, (9,7), source = 'local')

    data = data.dropna(subset=[data.columns[2]])
    total_col = int(data.iloc[1].last_valid_index())
    data = keep_cols_by_index(data,[1,total_col])
    data.columns = ['Category', 'Sales']
    data['Sales'] = data['Sales'].shift(-1)
    data = drop_na_by_name(data,['Category'])
    data = make_columns_numeric(data,['Sales'])

    if omega_loc:
        data['omega name'] = omega_client
        data['file date'] = file_date

    return data