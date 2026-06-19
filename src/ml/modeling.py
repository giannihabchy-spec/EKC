import pandas as pd
from supa.db import get_monthly_rates
from etl.utils import make_columns_date
from supa.db import get_omega_currency
from ml.config import (
    config_map,
    TRAIN_RATIO,
    VAL_RATIO
)
from supa.config import SHEET_CONFIG
from ml.config import LAGS, ROLLS



def add_old_data(sheets_dict):

    for df in sheets_dict.values():
        df['old_data'] = 1

    return sheets_dict


def match_monthly_rate(sheet):

    rates = get_monthly_rates()
    sheet_name, data = next(iter(sheet.items()))

    data = make_columns_date(data, ['report_date'])
    rates = make_columns_date(rates, ['date'])
    rates = rates.rename(columns = {'date': 'rates_date'})

    data = data.merge(
        rates[['rates_date','average_monthly_rate']],
        left_on='report_date',
        right_on='rates_date',
        how='left'
    ).rename(columns={'average_monthly_rate': 'client_rate'}).drop(columns='rates_date').copy()

    data['report_date'] = pd.to_datetime(data['report_date'])
    data.loc[data['report_date'].dt.year >= 2026, 'client_rate'] = 89000
    data = make_columns_date(data, ['report_date'])

    return {sheet_name: data}


def add_metadata(sheet, branch_id, supabase):
    sheet_name, data = next(iter(sheet.items()))

    currency = get_omega_currency(branch_id)['omega_currency']

    if not currency:
        return {
            'status': 'error',
            'msg': 'Currency is not available in Database'
        }

    data['branch_id'] = branch_id
    data['currency'] = currency
    data['old_data'] = 1

    return {
        'status': 'ok',
        'sheet': {sheet_name: data}
    }


def adjust_configs(prep_name):

    config = {
        key: value
        for key, value in SHEET_CONFIG.items()
        # if key in ["Sales", "Sales. Cat."]
        if key == config_map.get(prep_name)
    }

    if not config:
        return {
            'status': 'error',
            'msg': 'Bel amaliyye nzid l preprocessor 3al config_map'
        }

    for sheet_name, conf in config.items():
        conf['expected_columns'].append('old_data')

    return {
        'status': 'ok',
        'conf': config
    }


def convert_sheet_names_in_dict(sheet):

    return {
        config_map[key]: df
        for key, df in sheet.items()
        if key in config_map
    }


def get_series(df: pd.DataFrame, group_by: str = "item_group") -> dict[str, pd.Series]:
    """
    Split the flat DataFrame into one time series per group.

    group_by: "category" or "item_group"

    Returns:
        { group_name: pd.Series(sales, index=date, freq inferred) }
    """
    series = {}
    for name, group in df.groupby(group_by):
        s = (
            group.groupby("date")["sales"]
            .sum()
            .sort_index()
            .asfreq("D")           # daily frequency, NaN on missing days
            .fillna(0)
        ).round().astype(int)
        series[name] = s
    return series


def _split(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    n = len(s)
    if n < 20:
        raise ValueError(
            f"Series '{s.name}' has only {n} days — too short to split 70/15/15."
        )
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    return s.iloc[:train_end], s.iloc[train_end:val_end], s.iloc[val_end:]


def _make_features(s: pd.Series) -> pd.DataFrame:

    df = s.rename("sales").to_frame()

    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    for lag in LAGS:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    for window in ROLLS:
        df[f"roll_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"]  = df["sales"].shift(1).rolling(window).std()

    cols = [c for c in df.columns if c != "sales"]
    df = df.dropna(subset=cols)
    return df
