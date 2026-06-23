import pandas as pd
from etl.utils import make_columns_numeric


def merge_ib(cleaned: dict) -> dict:

    # disc_by_desc = cleaned.get("discount by description by employee")
    # disc_by_invoice = cleaned.get("discount by invoice with details")
    # disc_by_item = cleaned.get("discount by items")
    # cols = ["check", "description", "qty", "discount", "amount", "discount_percentage"]

    # if disc_by_desc is not None and disc_by_invoice is not None:
    #     disc_by_desc_100 = disc_by_desc[disc_by_desc["discount_percentage"] > 0.95].copy()
    #     cleaned["disc_by_desc__disc_by_invoice"] = disc_by_desc_100.merge(
    #         disc_by_invoice,
    #         how="left",
    #         on="check",
    #     )[cols]

    # if disc_by_desc is not None and disc_by_item is not None:
    #     disc_by_item__disc_by_desc = disc_by_item.merge(
    #         disc_by_desc,
    #         how="left",
    #         on="check",
    #     )[cols]
    #     disc_by_item__disc_by_desc = disc_by_item__disc_by_desc[
    #         disc_by_item__disc_by_desc["discount_percentage"] > 0.95
    #     ].copy()

    #     if disc_by_item__disc_by_desc.notna().any().any():

    #         ids = disc_by_item__disc_by_desc.index
    #         cleaned["disc_by_item__disc_by_desc"] = make_columns_numeric(
    #             disc_by_item__disc_by_desc,
    #             ["qty", "discount", "amount", "discount_percentage"],
    #         )
    #         cleaned["discount by items"] = cleaned["discount by items"].drop(index=ids)

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

    cols = ['check', 'description', 'qty', 'discount_percentage']

    desc = cleaned.get("discount by description by employee")
    inv = cleaned.get("discount by invoice with details")
    item = cleaned.get("discount by items") 

    if inv is None and item is None and desc is None:
        return cleaned


    elif inv is None or desc is None:

        item['discount_percentage'] = 1
        cleaned['final_discount'] = item[cols].copy()

        return cleaned


    elif item is None:

        cleaned['final_discount'] = inv.merge(
            desc[['check', 'discount_percentage']],
            on = 'check',
            how = 'left'
        )[cols].copy()

        return cleaned


    else:
        only_inv = inv[~inv['check'].isin(item['check'])]
        rest_inv = inv[inv['check'].isin(item['check'])]
        only_inv = only_inv.merge(
            desc[['check','discount_percentage']],
            on = 'check',
            how = 'left'
        )
        item = item.groupby(['check','description'], as_index=False).agg(
            {
                'qty': 'sum',
                'item_amount': 'sum',
                
            }
        )
        item_sum = item.groupby(['check'], as_index=False)['item_amount'].sum()
        x = item_sum.merge(
            desc[['check', 'discount']],
            on = 'check',
            how = 'left'
        )
        only_item_check = x.loc[x['item_amount'] == x['discount']].copy()
        only_item = item[item['check'].isin(only_item_check['check'])].copy()
        only_item['discount_percentage'] = 1
        only_item = only_item[cols].copy()
        doubles_check = x.loc[x['item_amount'] != x['discount']].copy()
        doubles_item = item[item['check'].isin(doubles_check['check'])]
        doubles_item = doubles_item.merge(
            desc[['check','amount']],
            on = 'check',
            how = 'left'
        ).drop_duplicates()
        doubles_item['discount_percentage'] = 1
        doubles_inv_check = inv[(inv['check'].isin(doubles_check['check'])) & ~(inv['description'].isin(doubles_item['description']))]
        doubles_inv_check = doubles_inv_check.merge(
            doubles_item[['check','item_amount','amount']],
            on = 'check',
            how = 'left'
        )
        doubles_inv_check['net_amount'] = doubles_inv_check['amount'] - doubles_inv_check['item_amount']
        doubles_desc = desc[desc['check'].isin(doubles_check['check'])]
        y = doubles_desc.merge(
            doubles_item[['check','item_amount']],
            on = 'check',
            how = 'left'
        )
        y = y[y['item_amount'] != y['discount']]
        doubles_inv = doubles_inv_check.merge(
            y[['check','discount']],
            on = 'check',
            how = 'left'
        )
        doubles_inv['discount_percentage'] = doubles_inv['discount'] / doubles_inv['net_amount']
        doubles_item = doubles_item[cols]
        doubles_inv = doubles_inv[cols]
        final = pd.concat([only_inv,only_item,doubles_inv,doubles_item], ignore_index=True)

        cleaned['final_discount'] = final

        return cleaned