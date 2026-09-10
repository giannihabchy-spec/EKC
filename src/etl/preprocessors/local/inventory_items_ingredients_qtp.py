from etl.utils import (
    read,
    keep_cols_by_index,
    drop_rows,
    drop_na_by_name,
    make_columns_numeric
)


def preprocess(path):
    data = read(path)
    data = keep_cols_by_index(data, [0,2,5,6])
    data.columns = ['code', 'product description', 'qty', 'unit']
    data = drop_rows(data, 'code', value = 'Product Code:')
    data = drop_na_by_name(data, ['code', 'product description'], 'all')
    data = data.reset_index(drop=True)
    category_mask = (
        data['product description'].isna()
        & data['product description'].shift(-1).isna()
        & data['product description'].shift(-2).isna()
    )
    group_mask = data['product description'].isna()
    data.loc[category_mask, 'category'] = data.loc[category_mask, 'code']
    data.loc[group_mask, 'group'] = data.loc[group_mask, 'code']
    data[['category', 'group']] = data[['category', 'group']].ffill()
    ids = data[data['code'].isna()].index
    data.loc[ids,'qty'] = data.loc[ids,'qty'].str.replace('Ingredients to prepare ','',regex=False)
    data.loc[ids,'production name'] = data.loc[ids,'qty'].str.split().apply(lambda x: ' '.join(x[3:]))
    data['production name'] = data['production name'].ffill()
    data.loc[ids,'to prepare'] = data.loc[ids,'qty'].str.split().apply(lambda x: x[:2])
    data.loc[ids,'qty to be prepared'] = data.loc[ids,'to prepare'].apply(lambda x: x[0])
    data.loc[ids,'prepared unit'] = data.loc[ids,'to prepare'].apply(lambda x: x[1])
    data[['qty to be prepared','prepared unit']] = data[['qty to be prepared','prepared unit']].ffill()
    data = data.drop(index = ids).copy()
    cols = [
    'category',
    'group',
    'production name',
    'product description',
    'qty',
    'unit',
    'qty to be prepared',
    'prepared unit'
    ]
    data = data[cols].copy()
    data.columns = ['category', 'group', 'production name', 'product description', 'qty', 'unit','qty to prepared', 'prepared unit']
    data = drop_na_by_name(data, ['production name', 'product description'], 'any')
    data = make_columns_numeric(data, ['qty', 'qty to prepared'])
    data = data.sort_values(['category', 'group', 'production name', 'product description'])
    return data





# def preprocess(path):
#     data = read(path)
#     data = keep_cols_by_index(data,[2,5,6])
#     data.columns = ['Product Description','Qty','Unit']
#     data = remove_repeated_headers(data,'Qty')
#     data = drop_na_by_name(data,['Product Description','Qty'])
#     data['get_ids'] = data['Qty']
#     data = make_columns_numeric(data,['get_ids'], er='coerce')
#     ids = data[data['get_ids'].isna()].index
#     # data.loc[ids,'Production Name'] = data.loc[ids,'Product Description']
#     # data['Production Name'] = data['Production Name'].ffill()
#     data.loc[ids,'Qty'] = data.loc[ids,'Qty'].str.replace('Ingredients to prepare ','',regex=False)
#     data.loc[ids,'Production Name'] = data.loc[ids,'Qty'].str.split().apply(lambda x: ' '.join(x[3:]))
#     data['Production Name'] = data['Production Name'].ffill()
#     data.loc[ids,'to prepare'] = data.loc[ids,'Qty'].str.split().apply(lambda x: x[:2])
#     data.loc[ids,'Qty to be Prepared'] = data.loc[ids,'to prepare'].apply(lambda x: x[0])
#     data.loc[ids,'Prepared Unit'] = data.loc[ids,'to prepare'].apply(lambda x: x[1])
#     data[['Qty to be Prepared','Prepared Unit']] = data[['Qty to be Prepared','Prepared Unit']].ffill()
#     data = data.drop(index=ids)
#     cols = ['Production Name', 'Product Description', 'Qty', 'Unit','Qty to be Prepared', 'Prepared Unit']
#     data = data[cols].copy()
#     data = make_columns_numeric(data,['Qty','Qty to be Prepared'])
#     data.columns = ['production name', 'product description', 'qty', 'unit','qty to prepared', 'prepared unit']
#     data = data.sort_values(['production name', 'product description'])
#     return data