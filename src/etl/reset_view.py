import xlwings as xw
from pathlib import Path
import streamlit as st

def workbook_is_open(wb_path):
    target = Path(wb_path).resolve()

    for app in xw.apps:
        for wb in app.books:
            try:
                if Path(wb.fullname).resolve() == target:
                    return True
            except Exception:
                pass

    return False


def reset_workbook_view(wb_path: str) -> None:

    if workbook_is_open(wb_path):
        st.error("Please close the Workbook 'Auto Calc.xlsx' and try again")
        st.stop()

    app = xw.App(visible=False, add_book=False)
    wb = None
    try:
        app.display_alerts = False
        app.screen_updating = False
        app.enable_events = False

        wb = app.books.open(wb_path)

        for sht in wb.sheets:
            api = sht.api

            try:
                if api.FilterMode:
                    api.ShowAllData()
            except Exception:
                pass
            try:
                api.AutoFilterMode = False
            except Exception:
                pass

            try:
                api.Columns.Hidden = False
            except Exception:
                pass
            try:
                api.Rows.Hidden = False
            except Exception:
                pass

        wb.save()
    finally:
        try:
            if wb is not None:
                wb.close()
        except Exception:
            pass
        app.quit()
