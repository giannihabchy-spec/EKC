@echo off
cd /d "%~dp0.."

uv run streamlit run "scripts/%~1" --theme.base=dark

pause