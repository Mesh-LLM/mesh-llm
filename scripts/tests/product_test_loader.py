"""Load product-owned tests through stable workspace unittest entrypoints."""
from importlib import util
from pathlib import Path
import sys


def load_product_tests(loader, product, filename):
    path = Path(__file__).resolve().parents[2] / product / "scripts/tests" / filename
    name = f"{product}_product_tests_{path.stem}"
    module = sys.modules.get(name)
    if module is None:
        spec = util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load product tests from {path}")
        module = util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return loader.loadTestsFromModule(module)
