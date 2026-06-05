from etl.utils import (
    read,
    keep_cols_by_index,
    drop_rows,
    drop_na_by_name,
    make_columns_numeric,
    make_columns_date,
    get_file_date,
    get_omega_client_name
)

def preprocess(path, omega_loc: bool = False):
    data = read(path)

    if omega_loc:
        omega_client = get_omega_client_name(data)
        file_date = get_file_date(data, (3,1))

    data = keep_cols_by_index(data, [0,1,2,3])
    data.columns = ['code', 'product', 'qty','total']

    cust_ids = data[data['code'].str.contains('Customer Name :', na = False)].index
    data.loc[cust_ids,'customer'] = data.loc[cust_ids,'product']
    data['customer'] = data['customer'].ffill()

    loc_ids = data[data['code'].str.contains('LOCATION NAME :', na = False)].index
    data.loc[loc_ids,'location'] = data.loc[loc_ids,'product']
    data['location'] = data['location'].ffill()

    date_ids = data[data['code'].str.contains('Sales Date :', na = False)].index
    data.loc[date_ids,'date'] = data.loc[date_ids,'product']
    data['date'] = data['date'].ffill()

    inv_ids = data[data['code'].str.contains('Invoice Number :', na = False)].index
    data.loc[inv_ids,'invoice number'] = data.loc[inv_ids,'product']
    data['invoice number'] = data['invoice number'].ffill()

    data = drop_rows(data, 'qty', 'Qty')
    data = drop_na_by_name(data, ['qty', 'total'])
    data = make_columns_numeric(data,['qty','total'])
    data = make_columns_date(data, ['date'])

    data['remarks'] = 'sales'
    cols = ['product', 'qty','total','customer','location','date','invoice number', 'remarks']
    data = data[cols]

    if omega_loc:
        data['omega name'] = omega_client
        data['file date'] = file_date

    return data