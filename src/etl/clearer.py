import xlwings as xw
from supa.modeling import normalize_column_name

def clear_all(
    master_path: str,
    jobs: list[dict],
    cleaned: dict | None = None,
) -> None:
    if cleaned is None:
        cleaned = {}

    app = xw.App(visible=False, add_book=False)
    try:
        wb = app.books.open(master_path)

        for job in jobs:

            sheet_name = job["sheet"]

            if sheet_name == "Recipes" and "sales items ingredients" not in cleaned:
                continue

            if sheet_name == "sub recipes" and "inventory items ingredients" not in cleaned:
                continue

            sht = wb.sheets[sheet_name]
            start_row = job["start_row"]

            for col in job["excel_cols"]:
                last_used = sht.range(
                    f"{col}{sht.cells.last_cell.row}"
                ).end("up").row

                if last_used >= start_row:
                    sht.range(
                        f"{col}{start_row}:{col}{last_used}"
                    ).value = None



        wb.save()
        wb.close()

    finally:
        app.quit()


INVALID_VALUES = {"", "#N/A", "#NA", "N/A", "nan", "None"}


def is_invalid(value) -> bool:
    if value is None:
        return True

    text = str(value).strip()
    return text in INVALID_VALUES


def clear_junk_rows(master_path: str, no_nulls: dict[str, list[str]]) -> None:
    app = xw.App(visible=False, add_book=False)
    app.display_alerts = False
    app.screen_updating = False
    app.enable_events = False

    try:
        wb = app.books.open(master_path)

        for sht in wb.sheets:
            if sht.name not in no_nulls:
                continue

            print(f"Processing: {sht.name}")

            if sht.api.ProtectContents:
                print(f"{sht.name}: protected, skipping")
                continue

            required_cols = [
                normalize_column_name(c)
                for c in no_nulls[sht.name]
            ]

            # -------------------------
            # CASE 1: SHEET HAS TABLE
            # -------------------------
            if sht.api.ListObjects.Count > 0:
                table = sht.api.ListObjects(1)

                if table.DataBodyRange is None:
                    print(f"{sht.name}: empty table")
                    continue

                try:
                    if table.AutoFilter.FilterMode:
                        table.AutoFilter.ShowAllData()
                except Exception:
                    pass

                headers = [
                    normalize_column_name(c)
                    for c in table.HeaderRowRange.Value[0]
                ]

                col_indexes = [
                    i + 1
                    for i, col in enumerate(headers)
                    if col in required_cols
                ]

                if not col_indexes:
                    print(f"{sht.name}: no matching columns")
                    continue

                row_count = table.ListRows.Count
                first_bad_row = None

                for table_row in range(1, row_count + 1):
                    for col_index in col_indexes:
                        value = table.DataBodyRange.Cells(
                            table_row,
                            col_index
                        ).Value

                        if is_invalid(value):
                            first_bad_row = table_row
                            break

                    if first_bad_row is not None:
                        break

                if first_bad_row is None:
                    print(f"{sht.name}: nothing to delete")
                    continue

                print(
                    f"{sht.name}: first invalid table row "
                    f"{first_bad_row}; deleting rows below"
                )

                for table_row in range(row_count, first_bad_row - 1, -1):
                    table.ListRows(table_row).Delete()

            # -------------------------
            # CASE 2: NORMAL SHEET
            # -------------------------
            else:
                used = sht.used_range

                header_row = used.row
                first_data_row = header_row + 1
                last_row = used.last_cell.row
                last_col = used.last_cell.column

                headers = sht.range(
                    (header_row, used.column),
                    (header_row, last_col)
                ).value

                if not isinstance(headers, list):
                    headers = [headers]

                headers = [
                    normalize_column_name(c)
                    for c in headers
                ]

                col_numbers = [
                    used.column + i
                    for i, col in enumerate(headers)
                    if col in required_cols
                ]

                if not col_numbers:
                    print(f"{sht.name}: no matching columns")
                    continue

                first_bad_row = None

                for excel_row in range(first_data_row, last_row + 1):
                    for col_number in col_numbers:
                        value = sht.cells(
                            excel_row,
                            col_number
                        ).value

                        if is_invalid(value):
                            first_bad_row = excel_row
                            break

                    if first_bad_row is not None:
                        break

                if first_bad_row is None:
                    print(f"{sht.name}: nothing to delete")
                    continue

                print(
                    f"{sht.name}: first invalid sheet row "
                    f"{first_bad_row}; deleting rows below"
                )

                sht.range(
                    f"{first_bad_row}:{last_row}"
                ).delete()

        wb.save()
        wb.close()

    finally:
        app.enable_events = True
        app.quit()