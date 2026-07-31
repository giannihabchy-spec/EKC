from etl.utils import (
    read,
    keep_cols_by_index, 
    drop_rows,
    drop_na_by_name,
    make_columns_numeric
)


def preprocess(path):
    data = read(path)
    data = keep_cols_by_index(data, [0, 1, 3, 4, 5, 6, 8, 11])
    data.columns = ['Product Code','Product Description','Pur Unit','Qty Pur','Inv Unit','Qty I F','Unit','Avg Cost']
    data = data.iloc[4:,].copy()
    data['Category'] = data.iloc[0,0]
    data = data.iloc[2:,].copy()
    data[data['Product Description'].isna()]
    data = drop_rows(data, 'Product Code', value = 'Product Code')
    data = drop_rows(data, 'Product Code', date = True)
    data.loc[data['Product Description'].isna(), 'Group'] = data.loc[data['Product Description'].isna(), 'Product Code']
    data['Group'] = data['Group'].ffill()
    data = drop_na_by_name(data, ['Product Description'])
    cols = ['Category','Group','Product Description','Qty I F','Unit','Pur Unit','Qty Pur','Inv Unit','Avg Cost','Product Code']
    data = data[cols]
    data = make_columns_numeric(data,['Qty I F','Qty Pur','Avg Cost'])
    data.columns = ['category','group','product description','qty I F','unit','pur unit','qty pur','inv unit','lbp','product code']
    return data


# def preprocess(path):
#     data = read(path)
#     data = keep_cols_by_index(data,[0, 1, 2, 3, 5, 6, 7, 9, 12])
#     data.columns = ['Item Id','Product Code','Product Description','Pur Unit','Qty Pur','Inv Unit','Qty I F','Unit','Avg Cost']
#     data = data.iloc[3:-1].copy()
#     data = remove_repeated_headers(data,'Product Code')
#     data = drop_rows(data,'Item Id',date = True)

#     # fill by pattern 
#     mask = (
#         data['Product Code'].isna() &
#         data['Product Code'].shift(-1).isna() &
#         data['Product Code'].shift(-2).isna()
#         )
#     data.loc[mask,'Category'] = data.loc[mask,'Item Id']
#     data['Category'] = data['Category'].ffill()

#     is_nan = data['Product Code'].isna()
#     end = is_nan & ~is_nan.shift(-1, fill_value=False)
#     ids = data.loc[end].index
#     data.loc[ids, 'Group'] = data.loc[ids,'Item Id']
#     data['Group'] = data['Group'].ffill()

#     data = drop_na_by_name(data,['Product Description'])
#     data = data.drop(['Item Id'], axis = 1)    
#     cols = ['Category','Group','Product Description','Qty I F','Unit','Pur Unit','Qty Pur','Inv Unit','Avg Cost','Product Code']
#     data = data[cols]
#     data = drop_na_by_name(data,['Unit'])
#     data = make_columns_numeric(data,['Qty I F','Qty Pur','Avg Cost'])
#     data.columns = ['category','group','product description','qty I F','unit','pur unit','qty pur','inv unit','lbp','product code']
#     return data