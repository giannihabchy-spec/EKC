import pandas as pd
from etl.utils import make_columns_date

info = {
    'Ending': {
        'sheet': 'Beg',
        'cols': ['location', 'product description', 'qty', 'unit', 'avg cost', 'total cost', 'month']
    },

    'Unit Cost': {
        'sheet': 'Rep. Variance',
        'cols': ['location', 'products']
    },

    'Recipes Eng': {
        'sheet': 'Rep.M.Eng.',
        'cols': ['menu items']
    },

    'Recipes Mix': {
        'sheet': 'Rep.M.Mix',
        'cols': ['menu items']
    },

    'Recipes Theo': {
        'sheet': 'Rep.Theo',
        'cols': ['menu items']
    },

}


def sheets_to_extract(cleaned_dict):
    sheets = ['Ending', 'Unit Cost', 'Recipes']
    mapping = {
        'Ending': 'Ending',
        'Unit Cost': 'programming summary inventory',
        'Recipes': 'sales items ingredients'
    }

    return [sht for sht in sheets if mapping[sht] not in cleaned_dict]


def extract_sheets(file_path, jobs, cleaned_dict):

    sheet_names = sheets_to_extract(cleaned_dict)

    with pd.ExcelFile(file_path) as xls:

        sheets_dict = {
            name: pd.read_excel(xls, sheet_name=name)
            for name in sheet_names
        }

        info_sht = pd.read_excel(xls, sheet_name='Info')
        locations = list(info_sht['Location'].dropna())
        
    if 'sales items ingredients' in cleaned_dict:
        recipes = cleaned_dict['sales items ingredients']
    else:
        recipes = sheets_dict.get('Recipes')

    if not recipes.empty:
        recipes.columns = [col_name.strip().lower() for col_name in recipes.columns]
        recipes['menu items'] = recipes['menu items'].drop_duplicates()
        recipes = recipes.dropna(subset=['menu items'])
        sheets_dict['Recipes Eng'] = recipes.copy()
        sheets_dict['Recipes Mix'] = recipes.copy()
        sheets_dict['Recipes Theo'] = recipes.copy()

        if 'Recipes' in sheets_dict:
            del sheets_dict['Recipes']


    if 'programming summary inventory' in cleaned_dict:
        uc = cleaned_dict['programming summary inventory']
    else:
        uc = sheets_dict.get('Unit Cost')

    if not uc.empty:
        uc.columns = [col_name.strip().lower() for col_name in uc.columns]
        uc = uc.loc[:,['product description']].copy()
        uc = uc.merge(
            pd.DataFrame({'location': locations}),
            how = 'cross'
        )
        uc = uc.rename(columns = {'product description': 'products'})
        uc = uc.drop_duplicates(subset = ['products', 'location']).copy()
        uc = uc.sort_values(['location', 'products']).copy()
        sheets_dict['Unit Cost'] = uc.copy()


    for key, data in sheets_dict.items():

        data.columns = [col_name.strip().lower() for col_name in data.columns]
        if 'month' in data.columns:
            data = make_columns_date(data,['month'])

        jobs.append(
            {
                'key': key,
                'df_cols': info.get(key)['cols'],
                'sheet': info.get(key)['sheet'],
                'start_row' : 2
            }
        )

        cleaned_dict[key] = data


    return {
        'jobs': jobs,
        'cleaned_dict': cleaned_dict
    }