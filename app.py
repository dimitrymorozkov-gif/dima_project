"""Default Streamlit entry point for the persistent group game."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("play.py")), run_name="__main__")
