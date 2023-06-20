import sys


def cleanup_modules():
    """Prevent test imports to cache modules since this will prevent mocking to work
        as expected.
    """
    sys_modules = list(sys.modules.keys())
    modules_to_reload = ["static_params", "edge_code", "edge_com", "server_code"]
    for imported_module in sys_modules:
        for component in modules_to_reload:
            if component in imported_module:
                del sys.modules[imported_module]
