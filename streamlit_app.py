"""
Entry point for Streamlit Community Cloud (and `streamlit run streamlit_app.py`).

Streamlit Cloud looks for `streamlit_app.py` in the repository root by default.
The dashboard itself lives in dashboard/app.py; this file simply runs it.
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py"),
               run_name="__main__")
