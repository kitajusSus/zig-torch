import ctypes
import pathlib
import platform

# Automatyczne wykrywanie rozszerzenia biblioteki
SYSTEM = platform.system()
EXT = {
    "Linux": ".so",
    "Darwin": ".dylib",
    "Windows": ".dll"
}[SYSTEM]

lib_path = pathlib.Path(__file__).parent.parent / "build" / f"libnative{EXT}"
lib = ctypes.CDLL(str(lib_path))

# Definicje typów
lib.tensor_add.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
]
lib.tensor_add.restype = None
