## Setup:
`uv sync`
<br><br>
## Requirements:
folder containg raw files + 'Auto Calc.xlsx'
<br><br>
## Terminal Run:
`uv run python scripts/run_etl.py 'folder_path'`  
(streamlit) `uv run streamlit run scripts/gui.py`
<br><br>
## Desktop shortcuts

All launchers use `shortcuts/run_app.bat` (hidden console via `.vbs`).

| Shortcut target | How you pick an app | What runs |
|-----------------|---------------------|-----------|
| `run_gui.vbs`, `run_recipes_gui.vbs`, … | One app per shortcut | Separate Streamlit process per app |
| **`run_launcher.vbs`** | Windows picker (`launcher.hta`) | One app at a time; new process each launch |
| **`run_hub.vbs`** | Streamlit sidebar pages | One process; switch apps in the browser |

Try **`run_launcher.vbs`** vs **`run_hub.vbs`** with two desktop shortcuts and see which you prefer.

- **Picker (HTA):** small native menu, each app is isolated, can open two apps in two browser tabs/windows.
- **Hub (multipage):** no extra window; navigation in Streamlit sidebar; shared session; only one server to stop.

Per-app `.vbs` files still work unchanged. To add an app to the picker, edit `shortcuts/launcher.hta`. For the hub, add `launcher/pages/N_Name.py` that calls `load_app("your_script.py")` (loads from `scripts/`).