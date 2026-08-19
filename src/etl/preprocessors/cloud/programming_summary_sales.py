import pandas as pd
import numpy as np
from etl.utils import (
    read,
    keep_cols_by_index,
    drop_na_by_name,
    make_columns_numeric,
    drop_rows
)


def preprocess(path):
    data = read(path)
    data = keep_cols_by_index(data, [0,1,4])
    data.columns = ['group', 'menu items', 'sp exc vat']
    ids = data.loc[data['group'] == 'Item ID'].index
    remove_ids = ids.union(ids-1).union(ids-2).union(ids-3)
    data = data.drop(index= remove_ids).reset_index(drop=True)
    data = drop_rows(data, 'menu items', value = 'Description')
    data = drop_rows(data, 'group', date = True)
    data = data.reset_index(drop = True)
    first_id = data[data['menu items'].notna()].index[0]
    data.iloc[first_id:, 0] = data.iloc[first_id:, 0].shift(-1)
    data['id'] = data['group']
    mask = (
        data['menu items'].isna() &
        data['menu items'].shift(-1).isna() & 
        data['menu items'].shift(-2).isna()
    )
    data.loc[mask, 'category'] = data.loc[mask, 'group']
    data.loc[data['menu items'].notna(), 'group'] = np.nan
    data[['category', 'group']] = data[['category', 'group']].ffill()
    data = drop_na_by_name(data, ['menu items'])
    data = make_columns_numeric(data, ['sp exc vat'])
    cols = ['id', 'category', 'group', 'menu items', 'sp exc vat']
    data = data[cols].copy()
    return data