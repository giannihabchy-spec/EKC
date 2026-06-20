import pandas as pd
from supa.db import get_branch_omega_name
from supa.validators import safe_ident
from etl.utils import make_columns_date
from supa.db import get_pg_connection

def validate_omega_name(sheets_dict, branch_id, supabase):
    supa_list = get_branch_omega_name(branch_id)['omega_name']

    data = next(iter(sheets_dict.values()))

    names = data['omega_name'].drop_duplicates()

    condition = (set(names).issubset(supa_list))
    if condition:
        return {
            'status': 'ok',
            'msg': 'Name checked'
        }
    
    return {
        'status': 'error',
        'msg': f"The files names '{set(names)}' do not match the selected client's Omega name '{supa_list}'"
    }


def validate_date(sheet):
    data = next(iter(sheet.values()))
    data = data[['report_date', 'file__name']].drop_duplicates()
    dups = []

    for i in data['report_date'].drop_duplicates():
        if len(data[data['report_date'] == i]) > 1:
            files = list(data.loc[data['report_date'] == i, 'file__name'])
            string_files = ', '.join(files)
            dups.append(f"Files '{string_files}' contain the same report date: '{str(i).split()[0]}'")
    
    if dups:
        return {
            'status': 'error',
            'msg': '    \n'.join(dups)
        }
    
    return {
        'status': 'ok',
        'msg': 'Dates Checked',
    }


def find_existing_data(conn, sheet, sheet_config, branch_id):
    conn.rollback()

    sht, data = next(iter(sheet.items()))

    # data = make_columns_date(data, ['report_date'])

    results = []
    table = sheet_config.get(sht)['target_table']
    table_name = safe_ident(table)

    dates = data['report_date'].drop_duplicates()
    
    for date in dates:
        year = date.year
        month = date.month
        start_date = date.replace(day=1)
        if month == 12:
            end_date = date.replace(year=year + 1, month=1, day=1)
        else:
            end_date = date.replace(month=month + 1, day=1)

        with conn.cursor() as cur:

            cur.execute(
                f"""
                select count(*)
                from {table_name}
                where branch_id = %s
                and report_date >= %s
                and report_date < %s
                """,
                (branch_id, start_date, end_date)
            )

            count = cur.fetchone()[0]

            if count > 0:
                results.append(f"{month}/{year} → {count} row(s)")

    if results:
        return {
            "status": "warning",
            "msg": "Data already exists in:  \n" + "  \n".join(results)
        }
    else:
        return {
            "status": "ok",
            "msg": "No existing data found."
        }
    

def delete_existing_data(conn, sheet, sheet_config, branch_id):
    conn.rollback()

    sht, data = next(iter(sheet.items()))

    # data = make_columns_date(data, ["report_date"])

    config = sheet_config.get(sht)
    if config is None:
        raise KeyError(
            f"No config found for sheet '{sht}'. "
            f"Available configs: {list(sheet_config.keys())}"
        )

    table = config["target_table"]
    table_name = safe_ident(table)

    dates = data["report_date"].drop_duplicates()
    deleted_rows = []

    try:
        for date in dates:
            year = date.year
            month = date.month

            start_date = date.replace(day=1)

            if month == 12:
                end_date = date.replace(year=year + 1, month=1, day=1)
            else:
                end_date = date.replace(month=month + 1, day=1)

            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {table_name}
                    WHERE branch_id = %s
                      AND report_date >= %s
                      AND report_date < %s
                      AND old_data = 1
                    """,
                    (branch_id, start_date, end_date)
                )

                if cur.rowcount > 0:
                    deleted_rows.append(
                        f"{month}/{year} → {cur.rowcount} row(s) deleted"
                    )

        conn.commit()

    except Exception as e:
        conn.rollback()
        return {
            "status": "error",
            "msg": f"Delete failed: {e}"
        }

    if deleted_rows:
        return {
            "status": "ok",
            "msg": "Deleted existing data from:  \n" + "  \n".join(deleted_rows)
        }

    return {
        "status": "warning",
        "msg": "Existing data is not old"
    }


def describe_series(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Quick summary of each series — useful for sanity-checking before modeling."""
    rows = []
    for name, s in series.items():
        rows.append({
            "group": name,
            "start": s.index.min().date(),
            "end": s.index.max().date(),
            "days": len(s),
            "zeros": (s == 0).sum(),
            "nulls": s.isna().sum(),
            "mean": round(s.mean(), 2),
            "std": round(s.std(), 2),
            "min": round(s.min(), 2),
            "max": round(s.max(), 2),
        })
    return pd.DataFrame(rows)


def delete_all_for_branch(branch_id, sheet, sheet_config): # for push_results


    sht, data = next(iter(sheet.items()))


    config = sheet_config.get(sht)
    if config is None:
        raise KeyError(
            f"No config found for sheet '{sht}'. "
            f"Available configs: {list(sheet_config.keys())}"
        )

    table = config["target_table"]
    table_name = safe_ident(table)

    deleted_rows = []
    conn = None

    try:
        conn = get_pg_connection()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                DELETE FROM {table_name}
                WHERE branch_id = %s
                """,
                (branch_id,)
            )

            if cur.rowcount > 0:
                deleted_rows.append(
                    f"{cur.rowcount} row(s) deleted"
                )

        conn.commit()

    except Exception as e:
        if conn:
            conn.rollback()

        return {
            "status": "error",
            "msg": f"Delete failed: {e}"
        }
    finally:
        if conn:
            conn.close()

    if deleted_rows:
        return {
            "status": "ok",
            "msg": "Deleted existing data from:  \n" + "  \n".join(deleted_rows)
        }

    return {
        "status": "ok",
        "msg": "No existing data found"
    }