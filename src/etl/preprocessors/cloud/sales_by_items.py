from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import drop_na_by_name
from etl.utils import remove_repeated_headers
from etl.utils import make_columns_numeric
from etl.utils import get_omega_client_name


def preprocess(path, omega_loc: bool = False):
    data = read(path)

    if omega_loc:
        omega_client = get_omega_client_name(data)

    data = keep_cols_by_index(data,[0,2,3])
    data.columns = ['description', 'qty sold', 'gross sales']
    data = drop_na_by_name(data,['qty sold'])
    data = remove_repeated_headers(data,'description')
    data = make_columns_numeric(data,['qty sold','gross sales'])

    if omega_loc:
        data['omega client name'] = omega_client
        cols = ['omega client name', 'description', 'qty sold']
        data = data[cols]

    return data