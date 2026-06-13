import pandas as pd
from etl.utils import (
    read,
    get_file_date,
    get_omega_client_name,
    drop_na_by_name,
    make_columns_date
)


def preprocess(path, omega_loc: bool = False):
    data = read(path)

    if omega_loc:
        omega_client = get_omega_client_name(data, (1,0))
        # file_date = get_file_date(data,(10,5),'local')

    data = data.iloc[14:-2, 1:-1].copy()
    data.iloc[0,0] = 'category'
    data.iloc[0,1] = 'group'
    data.iloc[0,2] = 'item'
    data.columns = data.iloc[0]
    data = data.iloc[1:].copy()
    data['category'] = data['category'].ffill()
    data[['category','group']] = data[['category','group']].shift(1)
    data = drop_na_by_name(data,['group'])
    data = data.drop(columns = ['item']).copy()
    data = data.melt(
        id_vars=["category", "group"],          # columns to keep
        var_name="date",                        # old column names
        value_name="sales"   
    )
    data['date'] = pd.to_datetime(data['date'], format='%d-%b-%y')
    data['report_date'] = data['date'].dt.to_period('M').dt.to_timestamp()
    data = make_columns_date(data,['report_date'])

    duplicates = data.duplicated(
        subset=['date', 'group'],
        keep=False
    )

    if duplicates.any():
        dup_rows = data.loc[duplicates, ['date', 'group']]
        raise ValueError(
            f"Non-unique date-group combinations found:\n"
            f"{dup_rows.drop_duplicates().to_string(index=False)}"
        )

    if omega_loc:
        data['omega name'] = omega_client
        # data['file date'] = file_date

    return data