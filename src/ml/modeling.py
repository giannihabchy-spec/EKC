from supa.db import get_monthly_rates
from etl.utils import make_columns_date
from supa.db import get_omega_currency
from ml.config import config_map
from supa.config import SHEET_CONFIG


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


def add_metadata(sheet, branch_id, supabase):
    sheet_name, data = next(iter(sheet.items()))

    currency = get_omega_currency(branch_id, supabase)['omega_currency']

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