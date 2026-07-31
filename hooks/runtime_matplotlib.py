"""PyInstaller runtime hook — enforce a safe matplotlib backend.

This hook runs *before* any application code so that ``matplotlib``
picks up ``MPLBACKEND`` before it reads potentially-problematic style
files (see Windows gbk-encoded locales).

Two Windows-specific hardening steps:

1. Force ``MPLBACKEND=QtAgg`` so the import chain never tries to spin up
   a GUI backend we do not use (avoids the earlier Windows encoding crash
   on import).

2. On Windows (especially CJK-locale systems like zh_CN.GBK), matplotlib's
   bundled ``mpl-data/stylelib/*.mplstyle`` files — or a user-level
   ``matplotlibrc``/style directory — may contain bytes that are not valid
   UTF-8 (e.g. a literal ``°`` written as a GBK byte 0xb0). matplotlib opens
   these files with ``encoding="utf-8"`` and *re-raises* the resulting
   ``UnicodeDecodeError``, which aborts the whole frozen app at import time.

   We patch ``matplotlib._open_file_or_url`` so local rc/style files are
   opened with ``errors="replace"``. Invalid bytes become the U+FFFD
   replacement char instead of crashing the app; rc keys we care about are
   unaffected.
"""
import os

os.environ.setdefault("MPLBACKEND", "QtAgg")


def _patch_matplotlib_file_open():
    import matplotlib

    _orig = matplotlib._open_file_or_url

    import contextlib
    import urllib.request

    @contextlib.contextmanager
    def _safe_open_file_or_url(fname):
        if (
            isinstance(fname, str)
            and fname.startswith(("http://", "https://", "ftp://", "file:"))
        ):
            ssl_ctx = matplotlib._get_ssl_context()
            with urllib.request.urlopen(fname, context=ssl_ctx) as f:
                yield (line.decode("utf-8", errors="replace") for line in f)
        else:
            fname = os.path.expanduser(fname)
            # The only change vs. upstream: errors="replace" so a non-UTF-8
            # style/rc file never crashes the frozen app.
            with open(fname, encoding="utf-8", errors="replace") as f:
                yield f

    matplotlib._open_file_or_url = _safe_open_file_or_url


_patch_matplotlib_file_open()
