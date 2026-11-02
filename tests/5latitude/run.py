from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def enable_windows_build_tools() -> None:
    """Populate this process with MSVC variables so gsplat can JIT its CUDA extension."""
    if os.name != "nt" or shutil.which("cl"):
        return
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
    ]
    script = next((p for p in candidates if p.is_file()), None)
    if script is None:
        raise RuntimeError("gsplat requires MSVC Build Tools, but VsDevCmd.bat was not found")
    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(script), buffer, len(buffer)):
        raise RuntimeError(f"Cannot resolve short path for {script}")
    command = f'{buffer.value} -arch=x64 -host_arch=x64 >nul && set'
    completed = subprocess.run(["cmd.exe", "/d", "/c", command], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(f"Failed to initialize MSVC Build Tools: {completed.stderr.strip()}")
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ[key] = value
    # Force ASCII/English compiler diagnostics; PyTorch's Windows probe decodes OEM output.
    os.environ["VSLANG"] = "1033"
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_root = str(Path(nvcc).resolve().parents[1])
        os.environ["CUDA_HOME"] = cuda_root
        os.environ["CUDA_PATH"] = cuda_root


enable_windows_build_tools()

from src.project import run_project


def main() -> int:
    try:
        result = run_project(ROOT)
        print(json.dumps({
            "status": "complete",
            "output": str(ROOT / "output"),
            "gaussian_count": result["inputs"]["gaussian"]["count"],
            "sh_degree": result["inputs"]["gaussian"]["active_sh_degree"],
            "subject_center": result["canonical"]["subject_center"],
            "sphere_radius": result["canonical"]["sphere_radius"],
            "validation_passed": result["validation"]["passed"]
        }, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
