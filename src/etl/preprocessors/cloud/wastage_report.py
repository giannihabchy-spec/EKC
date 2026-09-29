# remove grouping + not list view

from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import remove_repeated_headers
from etl.utils import drop_na_by_name
from etl.utils import make_columns_numeric, make_columns_date
from etl.utils import get_omega_client_name, get_file_date
from etl.utils import get_from_to_date_2

file_code = 'REP_I_00023D_rows'

def preprocess(path, omega_loc: bool = False):
    data = read(path)

    get_from_to_date_2(data, file_code, (3,3), (3,8))

    if omega_loc:
        omega_client = get_omega_client_name(data)
        file_date = get_file_date(data, [3,3])

    data = keep_cols_by_index(data,[0,3,4,10,13])
    data.columns = ['qty', 'product description', 'original remarks', 'date', 'location']
    data = drop_na_by_name(data,['product description'])
    data = remove_repeated_headers(data,'qty')
    cols = ['location','qty','product description','original remarks','date']
    data = data[cols]
    data = drop_na_by_name(data,['date'])
    data = make_columns_numeric(data,['qty'])
    data = make_columns_date(data,['date'])

    if omega_loc:
        data['omega name'] = omega_client
        data['file date'] = file_date

    return data