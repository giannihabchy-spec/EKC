import pandas as pd
from etl.utils import drop_na_by_name
from etl.utils import make_columns_date
from etl.utils import get_omega_client_name

def preprocess(path):
    data = pd.read_excel(path)
    omega_name = get_omega_client_name(data, (2,0))
    data = data.iloc[16:].copy()
    data.columns = (
        ["category", "group", "item"]
        + list(data.columns[3:])
    )
    data[['category', 'group']] = data[['category', 'group']].ffill()
    data.iloc[0,3:] = data.iloc[0,3:].bfill()
    data.iloc[1, 3:-1] = [
        str(cell).split()[0] if pd.notna(cell) else cell
        for cell in data.iloc[1, 3:-1]
    ]
    bad_cols = pd.to_numeric(data.iloc[1], errors="coerce").isna()

    bad_cols_names = data.columns[bad_cols][3:]
    data = data.drop(columns = bad_cols_names).copy()
    years = (
        data.iloc[0, 3:]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    months = (
        data.iloc[1, 3:]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    new_cols = (
        months.astype(float).astype(int).astype(str).str.zfill(2)
        + "-"
        + years.astype(float).astype(int).astype(str)
    )
    data.columns = ['category',
    'group',
    'item'] + list(new_cols)
    data = drop_na_by_name(data, ['item'])
    data = data.melt(
        id_vars=["category", "group", "item"],  # columns to keep
        var_name="date",                        # old column names
        value_name="qty"                        # values
    )
    data['date'] = pd.to_datetime(
        data['date'],
        format="%m-%Y",
        errors="raise"   # or "raise", "ignore"
    )
    data = make_columns_date(data, ['date'])
    data['omega_name'] = omega_name
    cols = ['omega_name','category', 'group', 'item', 'date', 'qty']
    data = data[cols]
    data.columns = ['omega_name','category', 'group', 'item', 'report date', 'qty sold']
    cols = ['omega_name', 'report date', 'category', 'group', 'item', 'qty sold']
    data = data[cols]
    return data