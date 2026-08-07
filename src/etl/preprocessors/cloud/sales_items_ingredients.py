# Price level 1, non separate pages, show cost

import pandas as pd
from etl.utils import read
from etl.utils import keep_cols_by_index
from etl.utils import remove_repeated_headers
from etl.utils import drop_rows
from etl.utils import drop_na_by_name
from etl.utils import make_columns_numeric


def preprocess(path):
    data = read(path)
    data = keep_cols_by_index(data,[0,1,3])
    data.columns = ['code','desc','qty']
    data = drop_rows(data, 'code', value = 'Product Code')
    data = drop_rows(data, 'code', date = True)
    data = data.reset_index(drop=True)
    ids = data[data['code'] == 'Price Level 1'].index - 1
    data.loc[ids,'item'] = data.loc[ids,'code']
    data['item'] = data['item'].ffill()
    data = drop_na_by_name(data, ['qty'])
    data = make_columns_numeric(data, ['qty'])
    cols = ['item','desc','qty']
    data = data[cols]
    data.columns = ['menu items','product description','qty']
    return data









# from etl.utils import (
#     read,
#     keep_cols_by_index,
#     drop_rows,
#     drop_na_by_name,
#     make_columns_numeric
# )

# def preprocess(path):
#     data = read(path)
#     data = keep_cols_by_index(data, [0,1,3,4])
#     data.columns = ['code', 'product description', 'qty', 'unit']
#     data = data.iloc[4:].copy()
#     data = drop_rows(data, 'code', 'Total by price level :')
#     data = drop_rows(data, 'code', date = True)
#     data = data.reset_index(drop= True)

#     items_ids = data[data['code'] == 'Price Level 1'].index - 1
#     for i in items_ids:
#         if data.loc[i,'code'] == 'Product Code':
#             data.loc[i,'menu items'] = data.loc[i-1,'code']
#         else:
#             data.loc[i,'menu items'] = data.loc[i,'code']

#     data['menu items'] = data['menu items'].ffill()
#     data = drop_rows(data, 'code', value = 'Price Level 1')
#     data = data.reset_index(drop=True)
#     mask = (
#         (data['code'] == 'Product Code')
#         & (data["code"].shift(-1) != data["menu items"].shift(-2))
#         & data["product description"].shift(-1).isna()
#     )

#     remove_ids = data.loc[mask].index + 1
#     data = data.drop(index = remove_ids).copy()
#     cat_mask = (
#         data['product description'].isna()
#         & data['product description'].shift(-1).isna()
#         & data['product description'].shift(-2).isna()
#         & data['product description'].shift(-3).isna()
#     )

#     data.loc[cat_mask, 'category'] = data.loc[cat_mask, 'code']
#     data['category'] = data['category'].ffill()
#     group_mask = (
#         data['product description'].isna()
#         & data['product description'].shift(-1).isna()
#     )

#     data.loc[group_mask, 'group'] = data.loc[group_mask, 'code']
#     data['group'] = data['group'].ffill()
#     data = drop_na_by_name(data, ['qty'])
#     data = drop_rows(data, 'qty', value = 'Qty')
#     cols = ['category', 'group', 'menu items', 'product description', 'qty', 'unit']
#     data = data[cols].copy()
#     data = make_columns_numeric(data, ['qty'])
#     return data
























