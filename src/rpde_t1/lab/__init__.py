"""Analysis instruments promoted from scripts/ (2026-09-02 cleanup).

These modules are imported by surviving scripts and by tests, so they live in the
package where a plain ``import rpde_t1.lab.<name>`` reaches them. Each keeps its CLI:
``uv run python -m rpde_t1.lab.<name> ...``. This ``__init__`` deliberately imports
nothing -- several submodules need torch or h5py, and the package root must stay cheap.
"""
