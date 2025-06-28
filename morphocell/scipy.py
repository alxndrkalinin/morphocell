"""Wraps SciPy submodules dynamically for device awareness."""

from functools import wraps
from importlib import import_module
from types import ModuleType
from typing import Any, Callable
import warnings

from .cuda import CUDAManager, asnumpy, to_device


class SciPyProxy(ModuleType):
    """Proxy module for dynamic wrapping of SciPy functions."""

    _loaded_modules: dict[str, "SciPyProxy"] = {}

    def __init__(self, name: str) -> None:
        """Initialize the proxy module."""
        super().__init__(name)
        self.cp = CUDAManager().get_cp()

    def __getattr__(self, func_name: str) -> Callable:
        """Dynamically wrap scipy or cupyx.scipy functions based on device."""
        if func_name in self.__dict__:
            return self.__dict__[func_name]

        def func_wrapper(*args: Any, **kwargs: Any) -> Any:
            array = args[0] if args else kwargs.get("input", None)
            use_gpu = False
            if self.cp is not None and hasattr(array, "device"):
                device_val = getattr(array, "device", None)
                use_gpu = hasattr(device_val, "id") or (isinstance(device_val, str) and device_val != "cpu")
            base_module = "cupyx.scipy" if use_gpu else "scipy"
            module_name = f"{base_module}.{self.__name__}"

            try:
                module = import_module(module_name)
                func = getattr(module, func_name)
            except (ModuleNotFoundError, AttributeError):
                warnings.warn(
                    f"cupyx.scipy.{self.__name__}.{func_name} is unavailable, falling back to CPU."
                )
                module = import_module(f"scipy.{self.__name__}")
                func = getattr(module, func_name)

                @wraps(func)
                def inner_cpu(*cargs: Any, **ckwargs: Any) -> Any:
                    cpu_args = [
                        asnumpy(a) if hasattr(a, "device") else a for a in cargs
                    ]
                    cpu_kwargs = {
                        k: asnumpy(v) if hasattr(v, "device") else v
                        for k, v in ckwargs.items()
                    }
                    result = func(*cpu_args, **cpu_kwargs)
                    if use_gpu:
                        if isinstance(result, tuple):
                            return tuple(
                                to_device(r, "GPU") if hasattr(r, "dtype") else r
                                for r in result
                            )
                        if hasattr(result, "dtype"):
                            return to_device(result, "GPU")
                    return result

                return inner_cpu(*args, **kwargs)

            @wraps(func)
            def inner(*cargs: Any, **ckwargs: Any) -> Any:
                return func(*cargs, **ckwargs)

            return inner(*args, **kwargs)

        self.__dict__[func_name] = func_wrapper
        return func_wrapper

    @classmethod
    def load_module(cls, name: str) -> "SciPyProxy":
        """Load the module if not already loaded."""
        if name not in cls._loaded_modules:
            cls._loaded_modules[name] = SciPyProxy(name)
        return cls._loaded_modules[name]


def __getattr__(name: str) -> SciPyProxy:
    """Load scipy proxy module."""
    return SciPyProxy.load_module(name)
