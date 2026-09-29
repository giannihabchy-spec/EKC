from etl.utils import read
from etl.utils import make_columns_numeric
from etl.utils import get_from_to_date_2


def preprocess(path):
    data = read(path)

    get_from_to_date_2(data, (11,6), (11,9), 'local')

    data = data.dropna(subset=[data.columns[2]])
    data = data.dropna(axis=1)
    data = data.iloc[:,[1,-1]].copy()
    data.columns=['Category', 'Total']
    data = make_columns_numeric(data,['Total'])
    data.columns = ['category', 'total']
    return data