"""CUDA kernels compiled at run time, with no compiler or build step at install.

The kernel source ships as a text file inside the package. NVRTC, which every
CUDA build of PyTorch already installs, compiles it the first time a projector
needs it; the CUDA driver API, part of the NVIDIA driver, loads the result and
launches it on PyTorch's current stream. Both are reached through ctypes, so
`pip install torchtomo` stays pure Python. Compiled binaries are cached on disk,
keyed by the source, the options, the NVRTC version, and the GPU architecture.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import hashlib
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

import torch


class KernelRuntimeError(RuntimeError):
    """NVRTC or the CUDA driver could not be loaded, compile, or launch."""


_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_void_pp = ctypes.POINTER(ctypes.c_void_p)
_c_size_p = ctypes.POINTER(ctypes.c_size_t)


def _torch_cuda_major() -> int | None:
    version = getattr(torch.version, "cuda", None)
    if not version:
        return None
    try:
        return int(version.split(".")[0])
    except ValueError:
        return None


def _nvidia_package_dirs(package: str, subdir: str) -> list[str]:
    """Library directories of a pip `nvidia-*` wheel, e.g. nvidia/cuda_nvrtc/lib."""
    try:
        module = __import__(f"nvidia.{package}", fromlist=["_"])
    except ImportError:
        return []
    paths = list(getattr(module, "__path__", []) or [])
    if getattr(module, "__file__", None):
        paths.append(os.path.dirname(module.__file__))
    return [os.path.join(path, subdir) for path in paths]


def _nvrtc_candidates() -> list[str]:
    """Places NVRTC can be, most specific first; the one matching PyTorch's CUDA wins."""
    override = os.environ.get("TORCHTOMO_NVRTC")
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    cuda_homes = [os.environ.get(key) for key in ("CUDA_HOME", "CUDA_PATH")] + ["/usr/local/cuda"]
    found: list[str] = []
    if sys.platform == "win32":
        dirs = [torch_lib, *_nvidia_package_dirs("cuda_nvrtc", "bin")]
        dirs += [os.path.join(home, "bin") for home in cuda_homes if home]
        for directory in dirs:
            found += sorted(glob.glob(os.path.join(directory, "nvrtc64_*.dll")), reverse=True)
        found.append("nvrtc64_120_0.dll")
        found.append("nvrtc64_112_0.dll")
    else:
        dirs = [*_nvidia_package_dirs("cuda_nvrtc", "lib"), torch_lib]
        if os.environ.get("CONDA_PREFIX"):
            dirs.append(os.path.join(os.environ["CONDA_PREFIX"], "lib"))
        dirs += [os.path.join(home, "lib64") for home in cuda_homes if home]
        for directory in dirs:
            names = glob.glob(os.path.join(directory, "libnvrtc*.so*"))
            found += sorted((name for name in names if "builtins" not in os.path.basename(name)), reverse=True)
        system = ctypes.util.find_library("nvrtc")
        if system:
            found.append(system)
        found += ["libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so"]
    major = _torch_cuda_major()
    if major is not None:
        # A stable sort keeps the search order within each group.
        found.sort(key=lambda path: 0 if _nvrtc_major_from_name(path) in (major, None) else 1)
    if override:
        found.insert(0, override)
    return list(dict.fromkeys(found))


def _nvrtc_major_from_name(path: str) -> int | None:
    name = os.path.basename(path)
    if name.startswith("nvrtc64_"):
        digits = name[len("nvrtc64_") :].split("_")[0]
        return int(digits[:-1]) if digits[:-1].isdigit() else None
    marker = ".so."
    if marker in name:
        head = name.split(marker, 1)[1].split(".")[0]
        return int(head) if head.isdigit() else None
    return None


def _driver_candidates() -> list[str]:
    if sys.platform == "win32":
        return ["nvcuda.dll"]
    return ["libcuda.so.1", "libcuda.so"]


def _bind(lib: ctypes.CDLL, name: str, argtypes, restype=ctypes.c_int, required: bool = True):
    try:
        function = getattr(lib, name)
    except AttributeError:
        if required:
            raise
        return None
    function.argtypes = argtypes
    function.restype = restype
    return function


@dataclass
class _Libraries:
    nvrtc: ctypes.CDLL
    driver: ctypes.CDLL
    nvrtc_path: str
    nvrtc_version: tuple[int, int]
    supported_archs: list[int] | None
    primary_contexts: dict = field(default_factory=dict)


_lock = threading.RLock()
_libraries: _Libraries | None = None
_load_error: str | None = None


def _open_nvrtc() -> tuple[ctypes.CDLL, str]:
    errors = []
    for candidate in _nvrtc_candidates():
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    detail = "; ".join(errors[-3:]) if errors else "no candidates"
    raise KernelRuntimeError(f"NVRTC not found ({detail}). Set TORCHTOMO_NVRTC to the library path.")


def _open_driver() -> ctypes.CDLL:
    for candidate in _driver_candidates():
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise KernelRuntimeError("CUDA driver library not found")


def _load() -> _Libraries:
    global _libraries, _load_error
    with _lock:
        if _libraries is not None:
            return _libraries
        if _load_error is not None:
            raise KernelRuntimeError(_load_error)
        try:
            _libraries = _load_uncached()
        except (KernelRuntimeError, OSError, AttributeError) as error:
            _load_error = str(error)
            raise KernelRuntimeError(_load_error) from error
        return _libraries


def _load_uncached() -> _Libraries:
    if getattr(torch.version, "hip", None):
        raise KernelRuntimeError("ROCm builds of PyTorch have no NVRTC")
    nvrtc, nvrtc_path = _open_nvrtc()
    driver = _open_driver()

    _bind(nvrtc, "nvrtcVersion", [_c_int_p, _c_int_p])
    _bind(nvrtc, "nvrtcGetErrorString", [ctypes.c_int], ctypes.c_char_p)
    _bind(
        nvrtc,
        "nvrtcCreateProgram",
        [_c_void_pp, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p],
    )
    _bind(nvrtc, "nvrtcCompileProgram", [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)])
    _bind(nvrtc, "nvrtcGetProgramLogSize", [ctypes.c_void_p, _c_size_p])
    _bind(nvrtc, "nvrtcGetProgramLog", [ctypes.c_void_p, ctypes.c_char_p])
    _bind(nvrtc, "nvrtcGetPTXSize", [ctypes.c_void_p, _c_size_p])
    _bind(nvrtc, "nvrtcGetPTX", [ctypes.c_void_p, ctypes.c_char_p])
    _bind(nvrtc, "nvrtcGetCUBINSize", [ctypes.c_void_p, _c_size_p], required=False)
    _bind(nvrtc, "nvrtcGetCUBIN", [ctypes.c_void_p, ctypes.c_char_p], required=False)
    _bind(nvrtc, "nvrtcDestroyProgram", [_c_void_pp])
    _bind(nvrtc, "nvrtcGetNumSupportedArchs", [_c_int_p], required=False)
    _bind(nvrtc, "nvrtcGetSupportedArchs", [_c_int_p], required=False)

    _bind(driver, "cuInit", [ctypes.c_uint])
    _bind(driver, "cuGetErrorName", [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)])
    _bind(driver, "cuDeviceGet", [_c_int_p, ctypes.c_int])
    _bind(driver, "cuDevicePrimaryCtxRetain", [_c_void_pp, ctypes.c_int])
    _bind(driver, "cuCtxGetCurrent", [_c_void_pp])
    _bind(driver, "cuCtxPushCurrent_v2", [ctypes.c_void_p])
    _bind(driver, "cuCtxPopCurrent_v2", [_c_void_pp])
    _bind(driver, "cuModuleLoadData", [_c_void_pp, ctypes.c_void_p])
    _bind(driver, "cuModuleGetFunction", [_c_void_pp, ctypes.c_void_p, ctypes.c_char_p])
    _bind(
        driver,
        "cuLaunchKernel",
        [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, _c_void_pp, _c_void_pp],
    )

    major, minor = ctypes.c_int(), ctypes.c_int()
    _nvrtc_check(nvrtc, nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)), "nvrtcVersion")
    archs = None
    if nvrtc.nvrtcGetNumSupportedArchs is not None and nvrtc.nvrtcGetSupportedArchs is not None:
        count = ctypes.c_int()
        if nvrtc.nvrtcGetNumSupportedArchs(ctypes.byref(count)) == 0 and count.value > 0:
            values = (ctypes.c_int * count.value)()
            if nvrtc.nvrtcGetSupportedArchs(values) == 0:
                archs = sorted(values)
    _driver_check(driver, driver.cuInit(0), "cuInit")
    return _Libraries(nvrtc, driver, nvrtc_path, (major.value, minor.value), archs)


def _nvrtc_check(nvrtc, code: int, what: str) -> None:
    if code != 0:
        message = nvrtc.nvrtcGetErrorString(code)
        raise KernelRuntimeError(f"{what}: {message.decode() if message else code}")


def _driver_check(driver, code: int, what: str) -> None:
    if code != 0:
        name = ctypes.c_char_p()
        driver.cuGetErrorName(code, ctypes.byref(name))
        raise KernelRuntimeError(f"{what}: {name.value.decode() if name.value else code}")


def runtime_available() -> bool:
    """True when NVRTC and the CUDA driver load. Cached; never raises."""
    if not torch.cuda.is_available():
        return False
    try:
        _load()
    except KernelRuntimeError:
        return False
    return True


def runtime_unavailable_reason() -> str | None:
    """Why runtime_available() is False, or None when it is True."""
    if not torch.cuda.is_available():
        return "CUDA is not available to PyTorch"
    try:
        _load()
    except KernelRuntimeError as error:
        return str(error)
    return None


def _cache_dir() -> str | None:
    override = os.environ.get("TORCHTOMO_KERNEL_CACHE")
    if override is not None:
        return override or None
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "torchtomo", "kernels")


def _read_cache(path: str | None) -> bytes | None:
    if path is None:
        return None
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def _write_cache(path: str | None, data: bytes) -> None:
    if path is None:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except OSError:
        pass


class _ContextGuard:
    """Make the device's primary context current, then restore whatever was current.

    PyTorch works in each device's primary context, so a module loaded there can
    use PyTorch's pointers and streams. Pushing only when needed leaves PyTorch's
    own notion of the current device untouched.
    """

    def __init__(self, libraries: _Libraries, index: int):
        self.driver = libraries.driver
        context = libraries.primary_contexts.get(index)
        if context is None:
            device = ctypes.c_int()
            _driver_check(self.driver, self.driver.cuDeviceGet(ctypes.byref(device), index), "cuDeviceGet")
            context = ctypes.c_void_p()
            _driver_check(
                self.driver,
                self.driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device.value),
                "cuDevicePrimaryCtxRetain",
            )
            libraries.primary_contexts[index] = context
        self.context = context
        self.pushed = False

    def __enter__(self):
        current = ctypes.c_void_p()
        _driver_check(self.driver, self.driver.cuCtxGetCurrent(ctypes.byref(current)), "cuCtxGetCurrent")
        if current.value != self.context.value:
            _driver_check(self.driver, self.driver.cuCtxPushCurrent_v2(self.context), "cuCtxPushCurrent")
            self.pushed = True
        return self

    def __exit__(self, *exc):
        if self.pushed:
            popped = ctypes.c_void_p()
            self.driver.cuCtxPopCurrent_v2(ctypes.byref(popped))
        return False


class KernelLibrary:
    """One CUDA source, compiled lazily per device, with named extern "C" kernels."""

    def __init__(self, source: str, name: str = "kernels.cu", options: tuple[str, ...] = ()):
        self.source = source
        self.name = name
        self.options = tuple(options)
        self._modules: dict[int, ctypes.c_void_p] = {}
        self._functions: dict[tuple[int, str], ctypes.c_void_p] = {}
        self.last_compile_seconds: float | None = None
        self.last_from_cache: bool | None = None

    def _target(self, libraries: _Libraries, index: int) -> tuple[str, bool]:
        """`sm_XY` for a cubin when NVRTC knows the device, otherwise PTX the driver can JIT."""
        major, minor = torch.cuda.get_device_capability(index)
        arch = major * 10 + minor
        archs = libraries.supported_archs
        can_cubin = libraries.nvrtc.nvrtcGetCUBIN is not None
        if archs is not None and arch in archs and can_cubin:
            return f"sm_{arch}", True
        if archs:
            lower = [value for value in archs if value <= arch]
            return f"compute_{max(lower) if lower else archs[0]}", False
        return f"compute_{arch}", False

    def _compile(self, libraries: _Libraries, target: str, cubin: bool) -> bytes:
        nvrtc = libraries.nvrtc
        program = ctypes.c_void_p()
        _nvrtc_check(
            nvrtc,
            nvrtc.nvrtcCreateProgram(ctypes.byref(program), self.source.encode(), self.name.encode(), 0, None, None),
            "nvrtcCreateProgram",
        )
        try:
            options = [f"--gpu-architecture={target}", "--std=c++17", *self.options]
            encoded = [option.encode() for option in options]
            array = (ctypes.c_char_p * len(encoded))(*encoded)
            result = nvrtc.nvrtcCompileProgram(program, len(encoded), array)
            if result != 0:
                size = ctypes.c_size_t()
                nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
                log = ctypes.create_string_buffer(max(size.value, 1))
                nvrtc.nvrtcGetProgramLog(program, log)
                raise KernelRuntimeError(f"NVRTC failed to compile {self.name}:\n{log.value.decode(errors='replace')}")
            size = ctypes.c_size_t()
            if cubin:
                _nvrtc_check(nvrtc, nvrtc.nvrtcGetCUBINSize(program, ctypes.byref(size)), "nvrtcGetCUBINSize")
                buffer = ctypes.create_string_buffer(size.value)
                _nvrtc_check(nvrtc, nvrtc.nvrtcGetCUBIN(program, buffer), "nvrtcGetCUBIN")
                return buffer.raw
            _nvrtc_check(nvrtc, nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)), "nvrtcGetPTXSize")
            buffer = ctypes.create_string_buffer(size.value)
            _nvrtc_check(nvrtc, nvrtc.nvrtcGetPTX(program, buffer), "nvrtcGetPTX")
            return buffer.raw
        finally:
            nvrtc.nvrtcDestroyProgram(ctypes.byref(program))

    def _module(self, libraries: _Libraries, index: int) -> ctypes.c_void_p:
        module = self._modules.get(index)
        if module is not None:
            return module
        start = time.perf_counter()
        target, cubin = self._target(libraries, index)
        key = hashlib.sha256(
            "\0".join([self.source, target, " ".join(self.options), "%d.%d" % libraries.nvrtc_version]).encode()
        ).hexdigest()[:32]
        directory = _cache_dir()
        path = os.path.join(directory, f"{key}.{'cubin' if cubin else 'ptx'}") if directory else None
        image = _read_cache(path)
        from_cache = image is not None
        module = ctypes.c_void_p()
        with _ContextGuard(libraries, index):
            if image is not None and libraries.driver.cuModuleLoadData(ctypes.byref(module), image) != 0:
                image = None
                from_cache = False
            if image is None:
                image = self._compile(libraries, target, cubin)
                _driver_check(
                    libraries.driver, libraries.driver.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData"
                )
                _write_cache(path, image)
        self._modules[index] = module
        self.last_compile_seconds = time.perf_counter() - start
        self.last_from_cache = from_cache
        return module

    def function(self, name: str, device: torch.device) -> "Kernel":
        index = device.index if device.index is not None else torch.cuda.current_device()
        key = (index, name)
        handle = self._functions.get(key)
        if handle is None:
            with _lock:
                handle = self._functions.get(key)
                if handle is None:
                    libraries = _load()
                    module = self._module(libraries, index)
                    handle = ctypes.c_void_p()
                    with _ContextGuard(libraries, index):
                        _driver_check(
                            libraries.driver,
                            libraries.driver.cuModuleGetFunction(ctypes.byref(handle), module, name.encode()),
                            f"cuModuleGetFunction({name})",
                        )
                    self._functions[key] = handle
        return Kernel(handle, name, index)


def _as_ctype(value):
    if value is None:
        return ctypes.c_void_p(None)
    if isinstance(value, torch.Tensor):
        return ctypes.c_void_p(value.data_ptr())
    if isinstance(value, ctypes._SimpleCData):
        return value
    if isinstance(value, bool):
        return ctypes.c_int(int(value))
    if isinstance(value, int):
        return ctypes.c_int(value)
    if isinstance(value, float):
        return ctypes.c_float(value)
    raise TypeError(f"unsupported kernel argument {type(value).__name__}")


class Kernel:
    """A loaded kernel. Launches go on PyTorch's current stream for the device."""

    __slots__ = ("handle", "name", "index")

    def __init__(self, handle: ctypes.c_void_p, name: str, index: int):
        self.handle = handle
        self.name = name
        self.index = index

    def __call__(self, grid: tuple[int, int, int], block: tuple[int, int, int], args, shared: int = 0) -> None:
        if min(grid) <= 0:
            return
        libraries = _load()
        values = [_as_ctype(value) for value in args]
        params = (ctypes.c_void_p * len(values))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in values])
        stream = ctypes.c_void_p(torch.cuda.current_stream(self.index).cuda_stream)
        with _ContextGuard(libraries, self.index):
            code = libraries.driver.cuLaunchKernel(self.handle, *grid, *block, shared, stream, params, None)
        _driver_check(libraries.driver, code, f"cuLaunchKernel({self.name})")
