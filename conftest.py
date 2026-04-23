"""Root conftest — registers the project as a proper package for relative imports.

The ComfyUI plugin root has __init__.py with relative imports that only work
when the directory is imported as part of a package. Pytest walks up from tests/
and finds this __init__.py, triggering ImportError. We pre-register the package
in sys.modules so the relative imports resolve correctly.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
ROOT_PKG = "pose_director_root"

# Register the project root as a proper package BEFORE pytest imports it
if ROOT_PKG not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        ROOT_PKG,
        str(PROJECT_ROOT / "__init__.py"),
        submodule_search_locations=[str(PROJECT_ROOT)],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[ROOT_PKG] = pkg
    # Also register under the directory name (with hyphens) that pytest may use
    sys.modules["comfyui-pose-director"] = pkg
    spec.loader.exec_module(pkg)

# Prevent pytest from collecting root __init__.py as a test module
collect_ignore = ["__init__.py", "nodes", "utils", "workflows"]
