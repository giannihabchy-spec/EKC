from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import drop_na_by_name
from etl.utils import remove_repeated_headers
from etl.utils import make_columns_numeric
from etl.utils import get_omega_client_name, get_file_date


def preprocess(path):
    data = read(path)

    omega_client = get_omega_client_name(data)
    file_date = get_file_date(data, [2,1])

    data = keep_cols_by_index(data,[0,2,3])
    data.columns = ['description', 'qty_sold', 'gross sales']
    data = drop_na_by_name(data,['qty_sold'])
    data = remove_repeated_headers(data,'description')
    data = make_columns_numeric(data,['qty_sold','gross sales'])

    data['omega_name'] = omega_client
    data['report_date'] = file_date
    cols = ['omega_name', 'report_date', 'description', 'qty_sold']
    data = data[cols]

    return data