import pandas as pd
import numpy as np
from etl.utils import (
    read,
    keep_cols_by_index,
    drop_na_by_name,
    make_columns_numeric
)


def preprocess(path):
    data = read(path)
    data = keep_cols_by_index(data, [0,1,5])
    data.columns = ['group', 'menu items', 'sp exc vat']
    # data['id'] = data['group']
    ids = data.loc[data['group'] == 'ID'].index
    remove_ids = ids.union(ids-1).union(ids-2).union(ids-3)
    data = data.drop(index= remove_ids).reset_index(drop=True)
    mask = (
        data['menu items'].isna() &
        data['menu items'].shift(-1).isna() & 
        data['menu items'].shift(-2).isna()
    )
    data.loc[mask, 'category'] = data.loc[mask, 'group']
    data.loc[data['menu items'].notna(), ['group']] = np.nan
    data[['group', 'category']] = data[['group', 'category']].ffill()
    data = drop_na_by_name(data, ['menu items'])
    cols = ['category', 'group', 'menu items', 'sp exc vat']
    # cols = ['id', 'category', 'group', 'menu items', 'sp exc vat']
    data = data[cols]
    data = make_columns_numeric(data, ['sp exc vat'])
    return data