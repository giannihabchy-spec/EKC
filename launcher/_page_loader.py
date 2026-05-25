"""Load a script from scripts/ inside the multipage hub."""
import importlib.util
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _ROOT / "scripts"
_SRC_DIR = _ROOT / "src"


def _ensure_paths() -> None:
    for entry in (str(_SCRIPTS_DIR), str(_SRC_DIR)):
        if entry not in sys.path:
            sys.path.append(entry)


def load_app(script_name: str) -> None:
    path = _SCRIPTS_DIR / script_name
    if not path.is_file():
        st.error(f"App script not found: {path}")
        return

    _ensure_paths()
    module_name = f"ekc_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        st.error(f"Could not load: {script_name}")
        return

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
