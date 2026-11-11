"""Alias entry point for app.py so 'streamlit run yourscript.py' works seamlessly."""
import runpy

if __name__ == "__main__":
    runpy.run_module("app", run_name="__main__")
