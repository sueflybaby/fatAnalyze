"""PyInstaller runtime hook — enforce a safe matplotlib backend.

This hook runs *before* any application code so that ``matplotlib``
picks up ``MPLBACKEND`` before it reads potentially-problematic style
files (see Windows gbk-encoded locales).
"""
import os

os.environ.setdefault("MPLBACKEND", "QtAgg")
