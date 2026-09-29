import pandas as pd
from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import make_columns_numeric
from etl.utils import validate_from_to_date

file_code = 'rep_i_0033'

def preprocess(path):
    data = read(path)

    to_date = pd.to_datetime(data.iloc[10,5])
    from_date = to_date.replace(day=1)
    validate_from_to_date(from_date, to_date, file_code)

    data = keep_cols_by_index(data,[2,4,5])
    data.columns = ['desc','loc','qty']
    loc_ids = data[data['loc']=='Location :'].index
    data.loc[loc_ids,'Location'] = data.loc[loc_ids,'qty']
    data['Location'] = data['Location'].ffill()
    data = data.drop(columns=['loc'])
    data.columns = ['Product Description', 'Qty', 'Location']
    data = data.dropna(how='any').copy()
    data = make_columns_numeric(data,['Qty'])
    data.columns = ['product description', 'qty', 'location']
    return data