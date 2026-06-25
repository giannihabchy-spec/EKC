import pandas as pd
import json
from supa.db import get_monthly_rates, get_pg_connection
from etl.utils import make_columns_date
from supa.db import get_omega_currency
from ml.config import (
    config_map,
    TRAIN_RATIO,
    VAL_RATIO
)
from supa.config import SHEET_CONFIG
from ml.config import (
    D_LAGS,
    D_ROLLS,
    W_LAGS,
    W_ROLLS,
)

_holidays_cache = None

HOLIDAY_TYPES = ["public_holiday", "government_holiday", "observance"]


def _load_holidays():
    global _holidays_cache
    if _holidays_cache is not None:
        return _holidays_cache

    conn = get_pg_connection()
    try:
        _holidays_cache = pd.read_sql("SELECT date, type FROM holidays", conn, parse_dates=["date"])
    finally:
        conn.close()

    _holidays_cache["type_key"] = (
        _holidays_cache["type"]
        .str.lower()
        .str.replace(" ", "_")
    )
    return _holidays_cache


def _add_holiday_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    holidays = _load_holidays()

    for type_key in HOLIDAY_TYPES:
        h_dates = set(holidays.loc[holidays["type_key"] == type_key, "date"].dt.normalize())

        if freq == "D":
            df[f"is_{type_key}"] = df.index.normalize().isin(h_dates).astype(int)
            pre_dates = {d - pd.Timedelta(days=1) for d in h_dates}
            post_dates = {d + pd.Timedelta(days=1) for d in h_dates}
            df[f"pre_{type_key}"] = df.index.normalize().isin(pre_dates).astype(int)
            df[f"post_{type_key}"] = df.index.normalize().isin(post_dates).astype(int)

            df[f"week_contain_{type_key}"] = 0
            df[f"week_contain_multiple_{type_key}"] = 0
            for d in h_dates:
                week_start = d - pd.Timedelta(days=d.weekday())
                week_end = week_start + pd.Timedelta(days=6)
                mask = (df.index.normalize() >= week_start) & (df.index.normalize() <= week_end)
                df.loc[mask, f"week_contain_{type_key}"] += 1
            df[f"week_contain_multiple_{type_key}"] = (df[f"week_contain_{type_key}"] > 1).astype(int)
            df[f"week_contain_{type_key}"] = (df[f"week_contain_{type_key}"] > 0).astype(int)

        else:
            df[f"contain_{type_key}"] = 0
            df[f"contain_multiple_{type_key}"] = 0
            for d in h_dates:
                mask = (df.index.normalize() <= d) & (d <= df.index.normalize() + pd.Timedelta(days=6))
                df.loc[mask, f"contain_{type_key}"] += 1
            df[f"contain_multiple_{type_key}"] = (df[f"contain_{type_key}"] > 1).astype(int)
            df[f"contain_{type_key}"] = (df[f"contain_{type_key}"] > 0).astype(int)

    return df



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


def get_series(
    df: pd.DataFrame,
    group_by: str = "category",
    freq: str = "D",
) -> dict[str, pd.Series]:

    series = {}

    series["total_sales"] = (
        df.groupby("date")["sales"]
        .sum()
        .sort_index()
        .resample(freq).sum()
        .asfreq(freq)
        .fillna(0)
        .round()
        .astype(int)
    )

    for name, group in df.groupby(group_by):
        s = (
            group.groupby("date")["sales"]
            .sum()
            .sort_index()
            .resample(freq).sum()
            .asfreq(freq)
            .fillna(0)
            .round()
            .astype(int)
        )

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


def _make_features(s: pd.Series, freq: str = "D" ) -> pd.DataFrame:

    df = s.rename("sales").to_frame()

    if freq == 'D':
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    df["month"] = df.index.month
    df["year"] = df.index.year
    df["week_of_month"] = ((df.index.day - 1) // 7 + 1)
    df["weekofyear"] = df.index.isocalendar().week.astype(int)

    if freq == 'D':
        for lag in D_LAGS:
            df[f"lag_{lag}"] = df["sales"].shift(lag)
        for window in D_ROLLS:
            df[f"roll_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
            df[f"roll_std_{window}"]  = df["sales"].shift(1).rolling(window).std()
    else:
        for lag in W_LAGS:
            df[f"lag_{lag}"] = df["sales"].shift(lag)  
        for window in W_ROLLS:
            df[f"roll_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
            df[f"roll_std_{window}"]  = df["sales"].shift(1).rolling(window).std()          


    df = _add_holiday_features(df, freq)

    cols = [c for c in df.columns if c != "sales"]
    df = df.dropna(subset=cols)
    return df


def result_to_json(model_name: str, result: dict) -> str:
    data = {
        "model":          model_name,
        "best_params":    result["best_params"],
        "metrics":        result["metrics"],
    }
    if result.get("final_features") is not None:
        data["final_features"] = result["final_features"]
    return json.dumps(data)