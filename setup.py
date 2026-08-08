"""Legacy setuptools entry point.

All distribution metadata is maintained exclusively in ``pyproject.toml``.
Keeping this file lets older build invocations continue to work without
creating a second, conflicting source of truth.
"""

from setuptools import setup

setup()
