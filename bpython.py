import os
import importlib
import sys

def _import_star(module_name, namespace=None):
    """Equivalent of 'from module_name import *'"""
    if namespace is None:
        namespace = globals()

    module = importlib.import_module(module_name)

    # Check if module defines __all__
    if hasattr(module, '__all__'):
        public_names = module.__all__
    else:
        # Otherwise, import all names not starting with _
        public_names = [name for name in dir(module) if not name.startswith('_')]

    for name in public_names:
        namespace[name] = getattr(module, name)


def __import_star(module_name, module_is_file=False):
    """Equivalent of 'from module_name import *'"""
    # Get the caller's frame
    caller_frame = sys._getframe(1)
    caller_globals = caller_frame.f_globals

    if module_is_file:
        # Convert file path to module name
        spec = importlib.util.spec_from_file_location("module.name", "/home/peter/dev/phd/slick/slick-measurements//repl.py")
        module = importlib.util.module_from_spec(spec)
    module = importlib.import_module(module_name)

    # Check if module defines __all__
    if hasattr(module, '__all__'):
        public_names = module.__all__
    else:
        # Otherwise, import all names not starting with _
        public_names = [name for name in dir(module) if not name.startswith('_')]

    for name in public_names:
        caller_globals[name] = getattr(module, name)


def import_star(file_path, call_depth=1):
    """Equivalent of 'from file import *' for a .py file"""
    # Get the caller's frame
    caller_frame = sys._getframe(call_depth)
    caller_globals = caller_frame.f_globals

    # Extract module name from file path (optional, but useful)
    module_name = file_path.replace('.py', '').replace('/', '_').replace('\\', '_')

    # Load the module from file
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check if module defines __all__
    if hasattr(module, '__all__'):
        public_names = module.__all__
    else:
        # Otherwise, import all names not starting with _
        public_names = [name for name in dir(module) if not name.startswith('_')]

    for name in public_names:
        caller_globals[name] = getattr(module, name)

def reload_repl(call_depth=2):
    import_star("repl.py", call_depth=call_depth)

def reload(call_depth=3):
    reload_repl(call_depth=call_depth)
