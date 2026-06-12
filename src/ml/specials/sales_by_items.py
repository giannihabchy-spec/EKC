from supa.db import (
    get_last_table
)
from etl.utils import drop_na_by_name


# def special_treatment(branch_id, sheet):

#     sp = get_last_sp(branch_id)
#     sheet_name, data = next(iter(sheet.items()))

#     data = data.merge(
#         sp[['menu_items', 'category', 'item_group']],
#         how = 'left',
#         left_on = 'description',
#         right_on = 'menu_items'
#     ).drop(columns='menu_items')

#     data = drop_na_by_name(data, ['category'])
#     data = data[data['gross_sales'] != 0].copy()
#     data = data[data['qty_sold'] != 0].copy()

#     return {sheet_name: data}


# def special_treatment(branch_id, sheet):

#     rec = get_last_table(branch_id, 'ac_recipes')
#     sheet_name, data = next(iter(sheet.items()))

#     data = data[data['description'].isin(rec['menu_items'])]
#     data = data[data['qty_sold'] != 0].copy()

#     return {sheet_name: data}


def special_treatment(branch_id, sheet):

    rec = get_last_table(branch_id, 'ac_recipes')
    sub = get_last_table(9,'ac_sub_recipes')
    sheet_name, data = next(iter(sheet.items()))

    data = data[data['description'].isin(rec['menu_items'])]
    data = data[data['qty_sold'] != 0].copy()


    cols = ['menu_items', 'product_description', 'qty', 'unit', 'qty_if']
    rec = rec[cols]
    rec = rec[rec['qty'] != 0].copy()

    res = rec.merge(
        data,
        'left',
        left_on= 'menu_items',
        right_on= 'description',
    ).drop(columns = ['description'])

    res = drop_na_by_name(res, ['qty_sold'])

    non_prod = res[~res['product_description'].isin(sub['production_name'])].copy()
    non_prod['stock_out'] = non_prod['qty'] * non_prod['qty_sold']

    cols = [
    'report_date',
    'branch_id',
    'product_description',
    'qty_if',
    'stock_out',
    'old_data' 
    ]
    non_prod = non_prod[cols].copy()

    prod = res[res['product_description'].isin(sub['production_name'])].copy()
    prod['qty_out'] = prod['qty'] * prod['qty_sold']

    res = sub.merge(
        prod,
        'left',
        left_on='production_name',
        right_on='product_description'
    ).drop(columns = ['product_description_y']).rename(columns={'product_description_x': 'product_description'})

    res['stock_out'] = (res['qty'] * res['qty_out']) / res['qty_to_prepared']

    prod['qty_out'] = prod['qty'] * prod['qty_sold']
    cols = [
    'product_description',
    'qty_if',
    'report_date',
    'branch_id',
    'old_data',
    'qty_out'
    ]
    prod = prod[cols]

    res = sub.merge(
        prod,
        'left',
        left_on='production_name',
        right_on='product_description'
    ).drop(columns = ['product_description_y']).rename(columns={'product_description_x': 'product_description'})

    res['stock_out'] = (res['qty'] * res['qty_out']) / res['qty_to_prepared']
    cols = [
    'report_date',
    'branch_id',
    'product_description',
    'qty_if',
    'stock_out',
    'old_data' 
    ]
    prod = res[cols]

    consumption = pd.concat([non_prod, prod])

    cons = consumption.groupby(['report_date','product_description'], as_index=False).agg(
        {
            'branch_id': 'first',
            'qty_if': 'first',
            'stock_out': 'sum',
            'old_data': 'first'
        }
    )
    cons['inv_stock_out'] = cons['stock_out'] / cons['qty_if']

    cons = cons[~cons['product_description'].isin(sub['production_name'])]

    return {
        sheet_name: data,
        'consumption': cons
    }