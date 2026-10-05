"""Workspace discovery entrypoint for product-owned tests."""
from scripts.tests.product_test_loader import load_product_tests


def load_tests(loader, tests, pattern):
    return load_product_tests(loader, 'skippy', 'test_package_native_runtime.py')
