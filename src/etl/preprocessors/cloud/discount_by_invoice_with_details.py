from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import drop_na_by_name
from etl.utils import drop_rows
from etl.utils import remove_repeated_headers
from etl.utils import make_columns_numeric
from etl.utils import clean_check
from etl.utils import get_from_to_date_1

file_code = 'REP_S_00513'

def preprocess(path):
    data = read(path)

    get_from_to_date_1(data, (2,5))

    data = keep_cols_by_index(data,[2,4,10])
    data.columns = ['Description', 'Check', 'QTY']
    data = clean_check(data,['Check'])
    data['Check'] = data['Check'].ffill()
    data = drop_na_by_name(data,['Description','QTY'])
    data = remove_repeated_headers(data,'Description')
    data = drop_rows(data,'Description',value='Invoice # :')
    data = data.iloc[:-2]
    data = make_columns_numeric(data,['QTY'])
    cols = ['Check', 'Description', 'QTY']
    data = data[cols]
    data.columns = ['check', 'description', 'qty']
    return data