import pandas as pd
from supa.modeling import normalize_column_name, clean_value
import numpy as np
from psycopg2.extras import execute_values
from psycopg2 import sql as psql
from supa.db import get_pg_connection
from etl.utils import make_columns_date


def load_sheet(file, sheet_name):
    df = pd.read_excel(file, sheet_name=sheet_name)
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df


def extract_info(file_path):

    with pd.ExcelFile(file_path) as xls:

        errors = []

        info_df = pd.read_excel(xls, sheet_name="Info")
        row = info_df.loc[0]

        file_date = row.get('Month')
        real_client = row.get('Restaurant Name')
        cur = row.get('Currency')
        if isinstance(cur,str):
            currency = (
                str(cur).strip()
                .replace("’", "")
                .replace("'", "")
                .title()
            )
        else:
            currency = pd.NA
        rate = row.get('Rate')

        info = {
            "status": 'ok',
            "msg": "All info extracted"
        }

        if pd.isna(file_date):
            errors.append('Invalid report date')

        if pd.isna(real_client):
            errors.append('Invalid client name')

        if pd.isna(currency):
            errors.append('Invalid currency')

        if pd.isna(rate):
            errors.append('Invalid rate')
        elif rate == 1:
            errors.append('⚠️ Rate = 1')

        if errors:
            info['status'] = 'error'
            info['msg'] = '  \n'.join(errors)

        return file_date, real_client, currency, rate, info


def extract_sheets_and_client(file_path, sheet_config):

    with pd.ExcelFile(file_path) as xls:

        errors = []

        common_names = [s for s in xls.sheet_names if s in sheet_config]
        sheets_dict = {
            name: pd.read_excel(xls, sheet_name=name)
            for name in common_names
        }
        
        info_df = pd.read_excel(xls, sheet_name="Info")
        row = info_df.loc[0]

        real_client = row.get('Restaurant Name')
        cur = row.get('Currency')
        if isinstance(cur,str):
            currency = (
                str(cur).strip()
                .replace("’", "")
                .replace("'", "")
                .title()
            )
        else:
            currency = pd.NA
            
        rate = row.get('Rate')

        excluded_sheets = {"Sales. Cat."}

        info = {
            "status": 'ok',
            "common_sheet_names": common_names,
            "missing_in_workbook": [s for s in sheet_config if s not in xls.sheet_names and s not in excluded_sheets],
            "extra_in_workbook": [s for s in xls.sheet_names if s not in sheet_config],
            "msg": "All info extracted"
        }

        if info['missing_in_workbook']:
            missing_str = (", ".join(info['missing_in_workbook']))
            errors.append('Missing sheets: ' + missing_str)

        if pd.isna(real_client):
            errors.append('Invalid client name')

        if pd.isna(currency):
            errors.append('Invalid currency')

        if pd.isna(rate):
            errors.append('Invalid rate')
        elif rate == 1:
            errors.append('⚠️ Rate = 1')

        if errors:
            info['status'] = 'error'
            info['msg'] = '  \n'.join(errors)

            return sheets_dict, real_client, currency, rate, info
        
        return sheets_dict, real_client, currency, rate, info


def push_sheets(sheets: dict, sheet_config: dict, conn, ingnore_missing_cols: bool = False):
    empty_sheets: list[str] = []
    loaded: list[str] = []

    def _fmt_lines(items: list[str]) -> str:
        return "  \n".join(items) if items else "None"

    try:
        with conn.cursor() as cur:
            for sheet_name, config in sheet_config.items():
                df = sheets.get(sheet_name)

                if df is None or not hasattr(df, "empty") or df.empty:
                    empty_sheets.append(sheet_name)
                    continue

                table = config["target_table"]
                expected_columns = config.get("expected_columns")
                unique_key = config.get("unique_key")

                if expected_columns:

                    cols_to_use = list(expected_columns)
                    if unique_key:
                        for c in unique_key:
                            if c not in cols_to_use:
                                cols_to_use.append(c)

                    for meta_col in ("branch_id", "report_date", "currency", "client_rate"):
                        if meta_col in df.columns and meta_col not in cols_to_use:
                            cols_to_use.append(meta_col)

                    missing = [c for c in cols_to_use if c not in df.columns]
                    if ingnore_missing_cols:
                        cols_to_use = [c for c in cols_to_use if c in df.columns]
                    else:
                        if missing:
                            raise ValueError(
                                f"Sheet '{sheet_name}' is missing required column(s): {', '.join(missing)}"
                            )
                    df = df[cols_to_use]

                rows = df.to_dict(orient="records")
                if not rows:
                    empty_sheets.append(sheet_name)
                    continue

                if unique_key:
                    missing_uk = [c for c in unique_key if c not in rows[0]]
                    if missing_uk:
                        raise ValueError(
                            f"Sheet '{sheet_name}' missing unique_key column(s): {', '.join(missing_uk)}"
                        )

                    seen = set()
                    for row in rows:
                        key = tuple(row.get(k) for k in unique_key)
                        if key in seen:
                            raise ValueError(f"Duplicate row in sheet '{sheet_name}' for unique key: {key}")
                        seen.add(key)

                cols = list(rows[0].keys())
                values = [[clean_value(row.get(c)) for c in cols] for row in rows]

                query = psql.SQL("INSERT INTO {table} ({cols}) VALUES %s").format(
                    table=psql.Identifier(table),
                    cols=psql.SQL(",").join(psql.Identifier(c) for c in cols),
                )

                try:
                    execute_values(cur, query, values)
                except Exception as e:
                    raise RuntimeError(f"Insert failed for sheet '{sheet_name}' into table '{table}': {e}") from e

                loaded.append(f"{sheet_name} → {len(values)} row(s)")

        conn.commit()
        return {
            "status": "ok",
            "message": (
                "Committed all sheets successfully.  \n  \n"
                f"Loaded:  \n{_fmt_lines(loaded)}  \n  \n"
                f"Skipped empty sheets:  \n{_fmt_lines(empty_sheets)}"
            ),
            "details": {"loaded": loaded, "empty": empty_sheets},
        }

    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass

        return {
            "status": "error",
            "message": (
                "Failed; rolled back everything  \n  \n"
                f"Reason:  \n{e}  \n  \n"
                f"Loaded before failure (not committed):  \n{_fmt_lines(loaded)}  \n  \n"
                f"Empty sheets:  \n{_fmt_lines(empty_sheets)}"
            ),
            "details": {"loaded": loaded, "empty": empty_sheets, "error": str(e)},
        }


def load_logs(branch_id, selected_client, selected_period, data_choice, client_currency, client_rate):
    data = {}

    table_mapping = {
        'waste:': 'waste_logs',
        'inventory:': 'inventory_logs',
        'transfers:': 'transfers',
        'production:': 'production_log',
        'purchase:': 'purchase_logs'
    }

    date = pd.to_datetime(selected_period)
    conn = get_pg_connection()

    try:

        if 'waste' in data_choice:

            first_day = date.to_period("M").start_time
            last_day = (date.to_period("M") + 1).start_time
            waste_query = """
            SELECT item_name, qty, remarks, date, item_type, location, created_at
            FROM waste_logs
            WHERE outlet = %s
            AND date >= %s
            AND date < %s
            """
            waste_df = pd.read_sql(
                waste_query, 
                conn, 
                params = (selected_client, first_day, last_day))
            waste_df.columns = ['product description', 'qty', 'original remarks', 'date', 'item type', 'location', 'created at']
            waste_df = make_columns_date(waste_df, ['created at'])

            data['waste_sales'] = waste_df.loc[waste_df['item type'] == 'Menu Items'].rename(columns = {'product description': 'product'}).drop(columns= 'item type').copy()
            data['waste_inventory'] = waste_df.loc[waste_df['item type'] == 'Inventory'].drop(columns= 'item type').copy()


        if 'inventory' in data_choice:

            first_day = date.to_period("M").end_time - pd.Timedelta(days=3)
            last_day = (date.to_period("M") + 1).start_time + pd.Timedelta(days=4)
            inv_query = """
            SELECT category, sub_category, item_name, quantity, location, date, created_at
            FROM inventory_logs
            WHERE outlet = %s
            AND date >= %s
            AND date < %s
            """
            inventory = pd.read_sql(
                inv_query,
                conn,
                params=(selected_client, first_day, last_day)
            )
            inventory.columns = ['category', 'group', 'product description', 'qty', 'location', 'date', 'created at']
            inventory = make_columns_date(inventory, ['created at'])

            data['inventory'] = inventory


        if 'production' in data_choice: 
            first_day = date.to_period("M").start_time
            last_day = (date.to_period("M") + 1).start_time
            prod_query = """
            SELECT production_name, actual_yield_qty, location, log_date, created_at
            FROM production_log
            WHERE log_date >= %s
            AND log_date < %s
            AND branch_id = %s
            """
            production = pd.read_sql(
                prod_query,
                conn,
                params=(first_day, last_day, branch_id)
            )
            production.columns = ['production list', 'qty', 'location', 'date', 'created at']
            production = make_columns_date(production, ['created at'])

            data['production'] = production


        if 'purchase' in data_choice:
            first_day = date.to_period("M").start_time
            last_day = (date.to_period("M") + 1).start_time
            query = """
            SELECT location, item_name, base_qty, sub_total, supplier_name, invoice_number, invoice_date, currency, created_at
            FROM purchase_logs
            WHERE outlet = %s
            AND invoice_date >= %s
            AND invoice_date < %s
            """
            purchase = pd.read_sql(
                query,
                conn,
                params=(selected_client, first_day, last_day)
            )
            purchase.columns = ['location', 'raw materials','qty','total cost','supplier names','invoice #','purchase date', 'currency', 'created at']
            purchase = make_columns_date(purchase, ['created at'])

            if client_currency == 'Usd':
                purchase.loc[purchase['currency'].str.lower() != client_currency.lower(), 'total cost'] = purchase.loc[purchase['currency'].str.lower() != client_currency.lower(), 'total cost'] / client_rate
            else:
                purchase.loc[purchase['currency'].str.lower() != client_currency.lower(), 'total cost'] = purchase.loc[purchase['currency'].str.lower() != client_currency.lower(), 'total cost'] * client_rate

            data['purchase'] = purchase


        if 'transfers' in data_choice:

            first_day = date.to_period("M").start_time
            last_day = (date.to_period("M") + 1).start_time
            transfers_query = """
            SELECT from_outlet, from_location, to_outlet, to_location, date, details, status, created_at
            FROM transfers
            WHERE date::timestamp >= %s
            AND date::timestamp < %s
            AND (
                from_outlet = %s
                OR to_outlet = %s
            )
            """
            trs = pd.read_sql(
                transfers_query,
                conn,
                params=(first_day, last_day, selected_client, selected_client)
            )

            trs = trs.explode('details')
            trs = trs.loc[trs['status'].isin(['Received', 'Received with Issue', 'Direct'])].copy()
            trs = make_columns_date(trs, ['date', 'created_at'])
            trs.columns = ['from branch','from location','to branch','to location','date','details','status', 'created at']
            trs['product'] = trs['details'].apply(lambda x: x.get('item_name'))
            trs['qty'] = trs['details'].apply(lambda x: x.get('received_qty'))
            trs['requested qty'] = trs['details'].apply(lambda x: x.get('requested_qty'))
            trs['fulfilled qty'] = trs['details'].apply(lambda x: x.get('fulfilled_qty'))
            trs['requested unit'] = trs['details'].apply(lambda x: x.get('requested_unit'))
            trs['fulfilled unit'] = trs['details'].apply(lambda x: x.get('fulfilled_unit'))
            trs = trs.drop(columns = 'details').copy()

            data['transfers'] = trs


    finally:
        conn.close()

    return {k:df for k,df in data.items() if not df.empty}