"""PyInstaller runtime hook — make VC++ redistributable DLLs discoverable.

On Windows, PyInstaller sometimes places ``MSVCP140.dll`` / ``VCRUNTIME140*.dll``
under a sub-package directory (e.g. ``shiboken6/`` or ``matplotlib.libs/``)
rather than next to the frozen ``.pyd`` files. Extension modules such as
``numpy.core._multiarray_umath`` then fail to load with
"Importing the numpy C-extensions failed", because the DLL search path does
not include those sub-directories.

Python 3.8+ honours ``os.add_dll_directory``, which extends the loader search
path for the current process. We register every directory under the frozen
bundle that contains a VC++ runtime DLL, so any ``.pyd`` can find it regardless
of where PyInstaller dropped it. This is the recommended fix for the
"DLL not found" class of frozen-app failures on Windows.
"""
import os
import sys


def _patch_vc_runtime_search_path():
    if sys.platform != "win32":
        return

    # Frozen apps have sys._MEIPASS pointing at the extracted bundle
    # (onedir: the directory containing the .exe). Fall back to the dir of
    # the executable.
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        base = os.path.dirname(sys.executable)

    markers = ("msvcp140", "vcruntime140", "msvcp140_1", "msvcp140_2")
    seen = set()
    added = []
    for root, dirs, files in os.walk(base):
        # Skip the top level we still add if it directly has the markers.
        if any(f.lower().startswith(m) and f.lower().endswith(".dll")
               for m in markers for f in files):
            if root not in seen:
                try:
                    os.add_dll_directory(root)
                    seen.add(root)
                    added.append(root)
                except (OSError, ValueError):
                    pass

    if added:
        os.environ.setdefault("BF_ANALYZER_DLL_DIRS", os.pathsep.join(added))


_patch_vc_runtime_search_path()
