import re


def safe_ident(name):
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


def find_existing_data(conn, sheet_config, branch_id, report_date):
    conn.rollback()

    results = []
    tables = [sht['target_table'] for sht in sheet_config.values()]

    start_date = f"{report_date}-01"
    
    year, month = map(int, report_date.split("-"))
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"

    with conn.cursor() as cur:
        for table in tables:
            table_name = safe_ident(table)

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
                results.append(f"{table} → {count} row(s)")

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


def delete_existing_data(conn, sheet_config, branch_id, report_date):
    conn.rollback()

    tables = [sht['target_table'] for sht in sheet_config.values()]

    start_date = f"{report_date}-01"
    
    year, month = map(int, report_date.split("-"))
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"

    deleted_tables = []

    try:
        with conn.cursor() as cur:
            for table in tables:
                table_name = safe_ident(table)

                cur.execute(
                    f"""
                    delete from {table_name}
                    where branch_id = %s
                      and report_date >= %s
                      and report_date < %s
                    """,
                    (branch_id, start_date, end_date)
                )

                if cur.rowcount > 0:
                    deleted_tables.append(f"{table} → {cur.rowcount} row(s) deleted")

        conn.commit()

        if deleted_tables:
            return {
                "status": "ok",
                "msg": "Deleted existing data from:  \n" + "  \n".join(deleted_tables)
            }
        else:
            return {
                "status": "warning",
                "msg": "No existing data found to delete."
            }

    except Exception as e:
        conn.rollback()
        return {
            "status": "error",
            "msg": f"Delete failed: {e}"
        }


def check_duplicates(SHEET_CONFIG, sheets):

    duplicates_report = {}

    for sheet_name, config in SHEET_CONFIG.items():

        if sheet_name not in sheets:
            continue

        data = sheets[sheet_name]
        unique_key = config.get("unique_key", [])

        if not unique_key:
            continue

        missing_cols = [col for col in unique_key if col not in data.columns]

        if missing_cols:
            continue

        duplicate_mask = data.duplicated(subset=unique_key, keep=False)
        duplicate_rows = data[duplicate_mask].sort_values(by=unique_key)

        if not duplicate_rows.empty:
            duplicates_report[sheet_name] = {
                "rows": duplicate_rows,
                "unique_key": unique_key
            }

    if duplicates_report:

        msgs = []

        for sheet_name, result in duplicates_report.items():

            duplicate_rows = result["rows"]
            unique_key = result.get("unique_key", [])

            if not unique_key:
                continue

            df_display = (
                duplicate_rows[unique_key]
                .drop_duplicates()
                .fillna("")
                .astype(str)
            )

            col_widths = {
                col: max(
                    df_display[col].astype(str).str.len().max(),
                    len(str(col))
                )
                for col in unique_key
            }

            header = " | ".join(
                str(col).ljust(col_widths[col]) for col in unique_key
            )

            separator = "-+-".join(
                "-" * col_widths[col] for col in unique_key
            )

            rows = [
                " | ".join(
                    str(row[col]).ljust(col_widths[col])
                    for col in unique_key
                )
                for _, row in df_display.iterrows()
            ]

            rows_text = header + "  \n" + separator + "  \n" + "  \n".join(rows)

            block = (
                f"⚠️ {sheet_name}: {len(duplicate_rows)} duplicate rows found  \n"
                f"{rows_text}"
            )

            msgs.append(block)

        return {
            "status": "error",
            "msg": "  \n  \n".join(msgs)
        }

    return {
        "status": "ok",
        "msg": "All constraints are satisfied"
    }


def check_rows(sheets, config):
    errors = []

    for sheet_name, sheet_config in config.items():
        if sheet_name not in sheets:
            continue

        data = sheets[sheet_name]
        cols = sheet_config.get("no_nulls")

        if not cols:
            continue

        mask = data[cols].isna() | (data[cols].astype(str).apply(lambda x: x.str.strip()) == "")

        bad_rows = mask.any(axis=1)

        if bad_rows.any():
            missing_cols = mask.loc[bad_rows].any(axis=0)
            missing_cols = missing_cols[missing_cols].index.tolist()

            errors.append(
                f"sheet '{sheet_name}': missing values in cols {missing_cols}"
            )

    if errors:
        return {
            "status": "error",
            "msg": "  \n".join(errors)
        }

    return {
        "status": "ok",
        "msg": "all good"
    }