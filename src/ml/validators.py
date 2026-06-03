from supa.db import get_branch_omega_name


def validate_omega_name(sheets_dict, branch_id, supabase):
    supa_list = get_branch_omega_name(branch_id, supabase)['omega_name']

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