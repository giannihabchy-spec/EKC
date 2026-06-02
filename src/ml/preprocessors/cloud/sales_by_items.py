from etl.preprocessors.cloud.sales_by_items import preprocess as original_prep
from supa.modeling import normalize_column_name


def preprocess(path):
    data = original_prep(path, omega_loc=True)
    data.columns = [normalize_column_name(col) for col in data.columns]
    data = data.rename(columns = {'file_date': 'report_date'})
    cols = ['description', 'qty_sold', 'gross_sales', 'omega_name', 'report_date']
    data = data[cols]
    return data