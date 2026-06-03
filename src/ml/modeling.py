from supa.db import get_monthly_rates
from etl.utils import make_columns_date


def add_old_data(sheets_dict):

    for df in sheets_dict.values():
        df['old_data'] = 1

    return sheets_dict


def match_monthly_rate(sheet):

    rates = get_monthly_rates()
    sheet_name, data = next(iter(sheet.items()))

    data = make_columns_date(data, ['report_date'])
    rates = make_columns_date(rates, ['date'])

    data = data.merge(
        rates[['date','average_monthly_rate']],
        left_on='report_date',
        right_on='date',
        how='left'
    ).rename(columns={'average_monthly_rate': 'client_rate'}).drop(columns='date').copy()

    return {sheet_name: data}