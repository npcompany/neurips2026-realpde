"""Imported automatically by every Python interpreter whose PYTHONPATH contains this directory
(`rpde_final.runtime` prepends it for every training stage; it runs the same code in-process).

Some mounted output volumes accept file data but refuse metadata changes: `chmod` and `utime`
fail with EPERM (seen on a RunPod network volume, 2026-10-05, where `shutil.copy2` of the
interval head's width rule aborted the `assemble` stage after 2.5 h of training). The training
chain copies files with `shutil.copy`, `copy2` and `copytree`, which call `copymode` /
`copystat` after writing the bytes. None of that metadata matters to the chain, so a failure to
set it is ignored here. The bytes are always copied first; a failed data copy still raises.
"""

import shutil

_copystat, _copymode = shutil.copystat, shutil.copymode


def _tolerant(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except PermissionError:
            return None

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


shutil.copystat = _tolerant(_copystat)
shutil.copymode = _tolerant(_copymode)
