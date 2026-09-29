from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import drop_na_by_name
from etl.utils import make_columns_numeric
from etl.utils import clean_check
from etl.utils import get_from_to_date_2

file_code = 'rep_s_00016'

def preprocess(path):
    data = read(path)

    get_from_to_date_2(data, file_code, (10,5), (10,7), 'local')

    data = keep_cols_by_index(data,[3,5,7,9])
    data.columns = ['Check','Description','QTY','amount']
    data = drop_na_by_name(data,['Check'])
    data = clean_check(data,['Check'])
    data = make_columns_numeric(data,['QTY','amount'])
    data.columns = ['check', 'description', 'qty', 'amount']
    data['discount percentage'] = 1
    return data