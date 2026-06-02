# from etl.utils import read
# from etl.utils import drop_na_by_name
# from etl.utils import make_columns_numeric
# from etl.utils import get_omega_client_name, get_file_date


# def preprocess(path):
#     data = read(path)

#     omega_client = get_omega_client_name(data)
#     file_date = get_file_date(data, [8,7], source = 'local')

#     data = data.iloc[11:].copy()
#     x = list(data.iloc[0])
#     x[0] = 'Description'
#     data.columns = x
#     cols = ['Description', 'lblTotal']
#     data = data[cols]
#     data.columns=['Description', 'Qty']
#     data = drop_na_by_name(data,['Qty'])
#     ids = data[data['Description'].isna()].index
#     data.loc[ids,'Total Amount'] = data.loc[ids,'Qty']
#     data['Total Amount'] = data['Total Amount'].shift(-1)
#     data = drop_na_by_name(data,['Description'])
#     data = make_columns_numeric(data,['Qty','Total Amount'])
#     data.columns = ['description', 'qty_sold', 'gross sales']

#     data['omega_name'] = omega_client
#     data['report_date'] = file_date
#     cols = ['omega_name', 'report_date', 'description', 'qty_sold']
#     data = data[cols]

#     return data


from etl.preprocessors.local.sales_by_menu_by_items import preprocess as original_prep
from supa.modeling import normalize_column_name

def preprocess(path):
    data = original_prep(path, omega_loc=True)
    data.columns = [normalize_column_name(col) for col in data.columns]
    data = data.rename(columns = {'file_date': 'report_date'})
    cols = ['description', 'qty_sold', 'gross_sales', 'omega_name', 'report_date']
    data = data[cols]
    return data