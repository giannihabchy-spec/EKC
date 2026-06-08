from supa.db import get_last_sp
from etl.utils import drop_na_by_name


def special_treatment(branch_id, sheet):

    sp = get_last_sp(branch_id)
    sheet_name, data = next(iter(sheet.items()))

    data = data.merge(
        sp[['menu_items', 'category', 'item_group']],
        how = 'left',
        left_on = 'description',
        right_on = 'menu_items'
    ).drop(columns='menu_items')

    data = drop_na_by_name(data, ['category'])
    data = data[data['gross_sales'] != 0].copy()
    data = data[data['qty_sold'] != 0].copy()

    return {sheet_name: data}