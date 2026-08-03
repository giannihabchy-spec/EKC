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


















# def merge_disc(cleaned: dict) -> dict:

#     desc = cleaned.get("discount by description by employee")
#     inv = cleaned.get("discount by invoice with details")

#     if desc is not None and inv is not None:
        
#         cleaned['discount by invoice with percentage'] = inv.merge(
#             desc[['check','discount percentage']],
#             on = 'check',
#             how = 'left'
#         )

#     final_inv = cleaned.get('discount by invoice with percentage')
#     item = cleaned.get('discount by items')

#     if final_inv is not None and item is not None:
#         cols = ['check', 'description', 'qty', 'discount percentage']
#         cleaned['final discount'] = pd.concat([
#             item[cols],
#             final_inv[cols]
#         ])

#     return cleaned