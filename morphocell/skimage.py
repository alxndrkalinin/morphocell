"""Wraps scikit-image submodules dynamically for device awareness."""
from importlib import import_module
from types import ModuleType
from functools import wraps

from .cuda import CUDAManager


class SkimageProxy(ModuleType):
    """Proxy module for dynamic wrapping of skimage functions for device awareness."""

    _loaded_modules: dict = {}

    def __init__(self, name):
        """Initialize the proxy module."""
        super().__init__(name)
        self.cp = CUDAManager().get_cp()

    def __getattr__(self, func_name):
        """Dynamically wrap skimage or cucim.skimage functions based on device capability."""
        if func_name in self.__dict__:
            return self.__dict__[func_name]

        def func_wrapper(*args, **kwargs):
            """Wrap skimage or cucim.skimage function based on device capability."""
            array = args[0] if args else kwargs.get("image", None)
            base_module = "skimage"

            if self.cp is not None and hasattr(array, "device"):
                base_module = "cucim.skimage"

            full_func_name = f"{base_module}.{self.__name__}.{func_name}"
            module_name, method_name = full_func_name.rsplit(".", maxsplit=1)
            module = import_module(module_name)
            func = getattr(module, method_name)

            @wraps(func)
            def inner_func(*args, **kwargs):
                """Inner function to call the wrapped function."""
                return func(*args, **kwargs)

            return inner_func(*args, **kwargs)

        self.__dict__[func_name] = func_wrapper
        return func_wrapper

    @classmethod
    def load_module(cls, name):
        """Load the module if not already loaded."""
        if name not in cls._loaded_modules:
            cls._loaded_modules[name] = SkimageProxy(name)
        return cls._loaded_modules[name]


def __getattr__(name):
    """Load skimage proxy module."""
    return SkimageProxy.load_module(name)
