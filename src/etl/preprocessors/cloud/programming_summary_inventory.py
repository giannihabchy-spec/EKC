from etl.utils import (
    read,
    keep_cols_by_index, 
    drop_rows,
    drop_na_by_name,
    make_columns_numeric
)


# def preprocess(path):     # by category
#     data = read(path)
#     data = keep_cols_by_index(data, [0,1,5,6,7,8,10,14])
#     data.columns = ['Product Code','Product Description','Pur Unit','Qty Pur','Inv Unit','Qty I F','Unit','Avg Cost']
#     data = data.iloc[4:,].copy()
#     data['Category'] = data.iloc[0,0]
#     data = data.iloc[2:,].copy()
#     data[data['Product Description'].isna()]
#     data = drop_rows(data, 'Product Code', value = 'Product Code')
#     data = drop_rows(data, 'Product Code', date = True)
#     data.loc[data['Product Description'].isna(), 'Group'] = data.loc[data['Product Description'].isna(), 'Product Code']
#     data['Group'] = data['Group'].ffill()
#     data = drop_na_by_name(data, ['Product Description'])
#     cols = ['Category','Group','Product Description','Qty I F','Unit','Pur Unit','Qty Pur','Inv Unit','Avg Cost','Product Code']
#     data = data[cols]
#     data = make_columns_numeric(data,['Qty I F','Qty Pur','Avg Cost'])
#     data.columns = ['category','group','product description','qty I F','unit','pur unit','qty pur','inv unit','lbp','product code']
#     data = drop_na_by_name(data, ['qty I F','unit','pur unit','qty pur','inv unit','lbp'])
#     data = data.sort_values(['category','group','product description'])
#     return data


def preprocess(path):     # all categories
    data = read(path)
    data = keep_cols_by_index(data, [0,1,5,6,7,8,10,14])
    data.columns = ['Product Code','Product Description','Pur Unit','Qty Pur','Inv Unit','Qty I F','Unit','Avg Cost']
    data = data.iloc[4:,].copy()

    cat_mask = (
        data['Product Description'].isna()
        & data['Product Description'].shift(-1).isna()
        & data['Product Description'].shift(-2).isna()
    )

    data.loc[cat_mask, 'Category'] = data.loc[cat_mask, 'Product Code']
    data['Category'] = data['Category'].ffill()


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
    data = drop_na_by_name(data, ['qty I F','unit','pur unit','qty pur','inv unit','lbp'])
    data = data.sort_values(['category','group','product description'])
    return data
