import pandas as pd
from etl.utils import make_columns_numeric


def merge_ib(cleaned: dict) -> dict:

    ib_parts: list[tuple[str, pd.DataFrame]] = []
    for name, df in cleaned.items():
        if isinstance(df, pd.DataFrame) and name.startswith("requisition summary IB "):
            ib_parts.append((name, df))

    if ib_parts:
        try:
            ib_parts_sorted = sorted(
                ib_parts,
                key=lambda x: int(x[0].split("requisition summary IB ")[1]),
            )
        except (IndexError, ValueError):
            ib_parts_sorted = ib_parts

        frames = [df for _, df in ib_parts_sorted]
        if frames:
            cleaned["requisition summary IB"] = pd.concat(frames, ignore_index=True)

    return cleaned


def merge_disc(cleaned: dict) -> dict:

    desc = cleaned.get("discount by description by employee")
    invoice = cleaned.get("discount by invoice with details")
    items = cleaned.get("discount by items")

    if desc is None and invoice is None and items is None:
        return cleaned
    


    elif items is not None and desc is None and invoice is None:

        cleaned['final discount'] = items
        return cleaned
    


    elif items is None and desc is not None and invoice is not None:

        # dups = desc.loc[desc['check'].duplicated()]
        # if not dups.empty:
        #     raise ValueError("duplicates in file 'discount by description' and discount by items not available.")

        desc = desc.groupby('check', as_index= False).agg(
            {
                'discount': 'sum',
                'amount': 'first',
                'discount percentage': 'sum'
            }
        )
        
        cleaned['final discount'] = invoice.merge(
            desc[['check', 'discount percentage']],
            on = 'check',
            how = 'left'
        )

        return cleaned



    elif items is not None and desc is not None and invoice is not None:

        item_checks = items['check'].drop_duplicates()
        inv = invoice.loc[~invoice['check'].isin(item_checks)]
        per = inv.merge(
            desc[['check', 'discount percentage']],
            on = 'check', 
            how = 'left'
        )
        items = items.drop(columns = 'amount').copy()
        final = pd.concat([items,per])

        cleaned['final discount'] = final

        return cleaned



    else:
        raise ValueError("Discount files not available in the right way.")


def merge_recipes(cleaned: dict, source: str) -> dict:

    recipes = cleaned.get("sales items ingredients")
    if source == 'cloud':
        sp = cleaned.get("programming summary sales")
    else:
        sp = cleaned.get("list sales items")

    if recipes is None or sp is None:
        return cleaned

    recipes = recipes.merge(
        sp[['menu items', 'category', 'group']],
        on = 'menu items',
        how = 'left'
    )[['category', 'group', 'menu items', 'product description', 'qty']].sort_values(
        ['category', 'group', 'menu items', 'product description']
    )

    cleaned["sales items ingredients"] = recipes
    return cleaned


def merge_sales_by_items(cleaned: dict, source: str) -> dict:

    sales = cleaned.get("sales by items")
    if source == 'cloud':
        sp = cleaned.get("programming summary sales")
    else:
        sp = cleaned.get("list sales items")

    if sales is None or sp is None:
        return cleaned

    sales = sales.merge(
        sp[['menu items', 'category', 'group']],
        left_on = 'description',
        right_on = 'menu items',
        how = 'left'
    )[['category', 'group', 'description', 'qty', 'gross sales']]

    sales[['category', 'group']] = sales[['category', 'group']].fillna('not available')

    cleaned["sales by items"] = sales
    return cleaned