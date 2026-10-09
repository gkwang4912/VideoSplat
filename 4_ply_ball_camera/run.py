from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def _prepend_path(path: Path) -> None:
    value = str(path)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if value.lower() not in {entry.lower() for entry in entries}:
        os.environ["PATH"] = value + os.pathsep + os.environ.get("PATH", "")


def _load_visual_studio_environment() -> None:
    """Load MSVC variables into this process without changing Python environments."""
    if shutil.which("cl.exe"):
        return
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"),
    ]
    vcvars = next((path for path in candidates if path.exists()), None)
    if vcvars is None:
        raise RuntimeError("Visual Studio 2022 C++ Build Tools were not found; gsplat cannot compile its CUDA extension.")
    completed = subprocess.run(
        f'call "{vcvars}" >nul && set',
        cwd=ROOT,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ[key] = value
    os.environ["VSLANG"] = "1033"
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"
    if not shutil.which("cl.exe"):
        raise RuntimeError("Visual Studio was found, but cl.exe was not added to PATH.")


def _configure_base_renderer() -> None:
    """Use CUDA installed in the currently running base Python prefix."""
    cuda_root = Path(sys.prefix) / "Library"
    nvcc = cuda_root / "bin" / "nvcc.exe"
    if not nvcc.exists():
        raise RuntimeError(
            f"CUDA compiler not found at {nvcc}. Run this project with the repaired base Python: {sys.executable}"
        )
    os.environ["CUDA_HOME"] = str(cuda_root)
    os.environ["CUDA_PATH"] = str(cuda_root)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("MAX_JOBS", str(min(os.cpu_count() or 1, 8)))
    _prepend_path(cuda_root / "bin")
    _prepend_path(cuda_root / "lib")
    _load_visual_studio_environment()
    # VS 2022 emits UTF-8 even when Python reports the legacy Windows OEM codec.
    # PyTorch otherwise fails before compilation while decoding `cl.exe` output.
    import torch.utils.cpp_extension as cpp_extension

    cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")


if __name__ == "__main__":
    _configure_base_renderer()
    sys.path.insert(0, str(ROOT))
    from src.pipeline import main

    main(ROOT)
