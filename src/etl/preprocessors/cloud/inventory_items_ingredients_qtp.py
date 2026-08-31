from etl.utils import (
    read,
    keep_cols_by_index,
    drop_na_by_name,
    remove_repeated_headers,
    make_columns_numeric,
    drop_rows
)


def preprocess(path):
    data = read(path)
    data = keep_cols_by_index(data,[0,1,2,4,5])
    data.columns = ['Code', 'Name', 'Product Description', 'Qty', 'Unit']
    data = drop_rows(data, 'code', value = 'Product Code')
    data = drop_rows(data, 'code', date = True)
    cat_mask = (
        data['Qty'].isna()
        & data['Qty'].shift(-1).isna()
        & data['Qty'].shift(-2).isna()
    )
    cat_ids = data.loc[cat_mask].index
    data.loc[cat_ids, 'Category'] = data.loc[cat_ids, 'Code']
    data['Category'] = data['Category'].ffill()
    group_ids = data[data['Qty'].isna()].index
    data.loc[group_ids, 'Group'] = data.loc[group_ids, 'Code']
    data['Group'] = data['Group'].ffill()
    ids = data[data['Name'].notna()].index
    data.loc[ids,'Qty'] = data.loc[ids,'Qty'].str.replace('Ingredients to prepare ','',regex=False)
    data.loc[ids,'Production Name'] = data.loc[ids,'Qty'].str.split().apply(lambda x: ' '.join(x[3:]))
    data['Production Name'] = data['Production Name'].ffill()
    data.loc[ids,'to prepare'] = data.loc[ids,'Qty'].str.split().apply(lambda x: x[:2])
    data.loc[ids,'Qty to be Prepared'] = data.loc[ids,'to prepare'].apply(lambda x: x[0])
    data.loc[ids,'Prepared Unit'] = data.loc[ids,'to prepare'].apply(lambda x: x[1])
    data[['Qty to be Prepared','Prepared Unit']] = data[['Qty to be Prepared','Prepared Unit']].ffill()
    data = drop_na_by_name(data,['Product Description','Qty'])
    data = make_columns_numeric(data,['Qty','Qty to be Prepared'])
    cols = ['Category', 'Group', 'Production Name', 'Product Description', 'Qty', 'Unit','Qty to be Prepared', 'Prepared Unit']
    data = data[cols].copy()
    data.columns = ['category', 'group', 'production name', 'product description', 'qty', 'unit','qty to prepared', 'prepared unit']
    data = data.sort_values(['category', 'group', 'production name', 'product description'])
    return data