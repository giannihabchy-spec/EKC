from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import drop_na_by_name
from etl.utils import drop_rows
from etl.utils import make_columns_numeric
from etl.utils import get_file_date
from etl.utils import get_omega_client_name


def preprocess(path, omega_loc: bool = False):
    data = read(path)

    if omega_loc:
        omega_client = get_omega_client_name(data)
        file_date = get_file_date(data, (2,2))

    total_col = int(data.iloc[3].last_valid_index())
    data = keep_cols_by_index(data,[0,total_col])
    data.columns = ['Category', 'Sales']
    data = drop_na_by_name(data,['Category','Sales'])
    data = drop_rows(data,'Category',value = 'Total ')
    data = make_columns_numeric(data,['Sales'])

    if omega_loc:
        data['omega_name'] = omega_client
        data['report_date'] = file_date

    return data