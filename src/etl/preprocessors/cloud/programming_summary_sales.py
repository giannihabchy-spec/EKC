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
    data = drop_rows(data, 'menu items', 'Description')
    data = drop_rows(data, 'group', 'Item Id')
    data = drop_rows(data, 'group', date = True)
    data = data.reset_index(drop = True)
    first_idx = data.loc[data['menu items'].notna()].index[0]
    data.loc[first_idx:, 'group'] = data.loc[first_idx:, 'group'].shift(-1)
    data = drop_na_by_name(data, list(data.columns), "all")
    data['id'] = data['group']
    group_mask = pd.to_numeric(data['id'], errors='coerce').notna()
    data.loc[group_mask, 'group'] = np.nan
    mask = (
        data['menu items'].isna() &
        data['menu items'].shift(-1).isna() &
        data['menu items'].shift(-2).isna()
    )
    data.loc[mask, 'category'] = data.loc[mask, 'group']
    data[['category', 'group']] = data[['category', 'group']].ffill()
    data = drop_na_by_name(data, ['menu items'])
    data = make_columns_numeric(data, ['sp exc vat'])
    cols = ['id', 'category', 'group', 'menu items', 'sp exc vat']
    data = data[cols].copy()
    return data