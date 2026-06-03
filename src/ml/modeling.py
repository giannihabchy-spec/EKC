def add_old_data(sheets_dict):

    for df in sheets_dict.values():
        df['old_data'] = 1

    return sheets_dict