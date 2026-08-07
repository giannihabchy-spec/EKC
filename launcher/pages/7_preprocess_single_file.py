import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _page_loader import load_app

load_app("preprocess_single_file.py")
