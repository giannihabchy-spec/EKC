from etl.orchestrator import cleaner_by_code


files_to_keep = {
    'REP_I_00268.xlsx': 'summary of sales by customer by item',
    'REP_I_00462.xlsx': 'purchase master report for all branches',
    'rep_s_00191_rows.xlsx': 'sales by items',
    'REP_I_00023D_rows.xlsx': 'wastage report',
    'REP_I_00074.xlsx': 'sales item wastage',
    'rep_i_0051.xls': 'purchase with all details',
    'rep_s_00138.xls': 'sales by menu by items',
    'rep_i_00268_s.xls': 'sales / summary of sales by customer by items',
    'rep_i_00268.xls': 'inventory / summary of sales by customer by items',
    'rep_i_0023.xls': 'inventory wastage items',
    'rep_i_0074.xls': 'sales wastage items'
}

def adjust_cleaners():
    files = set(files_to_keep.keys())

    return {
        source: {
            filename: value
            for filename, value in cleaners.items()
            if filename in files
        }
        for source, cleaners in cleaner_by_code.items()
    }