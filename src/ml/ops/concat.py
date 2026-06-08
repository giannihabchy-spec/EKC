from pathlib import Path
import pandas as pd
import os


def concat_files(folder_path, preprocessing_func):
    folder = Path(folder_path)

    xlsx_files = list(folder.glob("*.xlsx"))
    xls_files = list(folder.glob("*.xls"))

    if xlsx_files and xls_files:
        return {
            'status': 'error',
            'msg': 'Mixed file types'
        }
    

    excel_files = xlsx_files or xls_files

    if not excel_files:
        return {
            'status': 'error',
            'msg': 'Empty folder'
        }

    all_dfs = []

    for file_path in excel_files:
        df = preprocessing_func(file_path)
        file__name = os.path.basename(file_path)
        df['file__name'] = file__name
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    # combined_df = combined_df.sort_values(
    #     by="report_date",
    #     ascending=False
    # )

    name = preprocessing_func.__module__.split(".")[-1].replace("_", " ")
    final_name = f"{name}.xlsx"
    destination = folder.parent
    # combined_df.to_excel(output_path, index=False)

    return {
        'status': 'ok',
        'msg': 'Data is saved',
        'destination': destination,
        'prep_name': name,
        'final_name': final_name,
        'data': {name: combined_df}
    }