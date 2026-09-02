sheets_sorting = {
    'programming summary inventory': ['category','group','product description'],
    'programming summary sales': ['category', 'group', 'menu items'],
    'list sales items': ['category', 'group', 'menu items'],
    'sales items ingredients': ['menu items','product description'],
    'inventory items ingredients': ['category', 'group', 'production name', 'product description'],
}

def _sort(cleaned: dict) -> dict:

    for i in sheets_sorting:
        if i in cleaned:
            cleaned[i] = cleaned[i].sort_values(sheets_sorting[i])

    return cleaned