import pandas as pd
import os
from supabase import create_client as supabase_init
from pathlib import Path
import tomllib
from etl.utils import (
    read,
    drop_na_by_name,
    get_omega_client_name,
    make_columns_date
)
from supa.db import (
    get_last_table,
    get_branch_id_from_omega_name
)
from supa.modeling import normalize_string_columns
ROOT = Path(__file__).parents[4]
SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"
with SECRETS_PATH.open("rb") as f:
    secrets = tomllib.load(f)
for key in ["url", "key", "host", "port", "name", "user", "password"]:
    if key in secrets:
        os.environ[key] = str(secrets[key])  
supabase = supabase_init(secrets["url"], secrets["key"])



def preprocess(path, omega_loc: bool = False):
    data = read(path)

    omega_client = get_omega_client_name(data)

    branch_id = get_branch_id_from_omega_name(omega_client, supabase)['branch_id']
    sp = get_last_table(branch_id, 'ac_selling_prices').rename(columns = {'item_group': 'group'})

    data = data.iloc[3:-1,2:].copy().reset_index(drop=True)

    from_ids = data.loc[data[2].notna()].index

    mask = (
        data[4].notna() &
        pd.to_numeric(data[4], errors='coerce').isna()
    )

    date_ids = data.loc[mask].index
    unique_date_ids = data.loc[date_ids,4].drop_duplicates().index
    drop_date_ids = date_ids.difference(unique_date_ids)
    drop_ids = from_ids.union(drop_date_ids)

    data = data.drop(index=drop_ids).copy()

    splits = sorted(unique_date_ids)
    dfs = []
    for i, start in enumerate(splits):
        end = splits[i + 1] - 1 if i + 1 < len(splits) else data.index[-1]
        chunk = data.loc[start:end - 1].reset_index(drop=True)
        dfs.append(chunk)
    data = pd.concat(dfs, axis=1)

    data = data.iloc[:,1:].astype(object).copy()

    data.iloc[0] = pd.to_datetime(
        data.iloc[0].str.split().str[0],
        format='%d-%b-%Y' ,errors= 'coerce'
    ).dt.date

    data.iloc[0,0] = 'item'
    mask = data.iloc[0].notna()
    data.columns = data.iloc[0]
    data = data.iloc[1:].copy()
    data = data.loc[:, mask.values]
    data = data.reset_index(drop=True)

    data['item'] = data['item'].shift(1)
    data = drop_na_by_name(data, ['item'])

    data = normalize_string_columns({'data': data}).get('data').get('data')

    data = data.merge(
        sp[['menu_items','category','group']],
        'left',
        left_on = 'item',
        right_on = 'menu_items'
    ).drop(columns = ['menu_items','item']).copy()

    data = data.melt(
        id_vars=["category", "group"],          # columns to keep
        var_name="date",                        # old column names
        value_name="sales"  
    )

    data = (
        data.groupby(['group', 'date'], as_index=False)
        .agg({
            'sales': 'sum',
            'category': 'first'
        })
    )

    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
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
        data['omega_name'] = omega_client

    return data