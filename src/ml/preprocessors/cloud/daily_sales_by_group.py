from etl.preprocessors.cloud.daily_sales_by_group import preprocess as original_prep
from supa.modeling import normalize_column_name

def preprocess(path):
    data = original_prep(path, omega_loc=True)
    data.columns = [normalize_column_name(col) for col in data.columns]
    return data