"""Minimal setuptools shim for Aloha_Act so `pip install -e Aloha_Act` works.

Note: ace_agent.py in OSWorld imports ui_aloha via sys.path.insert() at
runtime, so this install is technically optional — the runner only needs
--aloha_path to point at this directory. But installing lets you `import
ui_aloha` from anywhere in the venv, which is handy for development.

Uses find_namespace_packages because the upstream tree doesn't ship
__init__.py at every level (implicit namespace packages, PEP 420).
"""

from setuptools import setup, find_namespace_packages

setup(
    name="ui-aloha",
    version="0.1.0",
    description="ShowUI-Aloha Act — computer-use agent framework (planner + client + guidance loader)",
    packages=find_namespace_packages(include=["ui_aloha", "ui_aloha.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    # Runtime deps are listed at repo root: ShowUI-Aloha/requirements.txt.
    # Not duplicated here to avoid drift; install them via
    # `pip install -r ../requirements.txt` before or after this editable
    # install.
    install_requires=[],
)
