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

Each app can have its own `.vbs` under `shortcuts/` (hidden console → `run_app.bat` → Streamlit).

**One shortcut for all apps:** point the desktop shortcut at `shortcuts/run_launcher.vbs`. It opens a small picker (`launcher.hta`) to choose Auto Calc, Recipes, or Push to Database.

To add another app later: add a button in `shortcuts/launcher.hta` and pass the script filename (same as the existing per-app `.vbs` files).