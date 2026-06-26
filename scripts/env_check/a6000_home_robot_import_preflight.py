#!/usr/bin/env python3
"""Import-only preflight for the A6000 HomeRobot/OVMM environment.

This script intentionally avoids simulator construction, rendering, dataset
downloads, and policy loading. It only checks Python package visibility and
CUDA visibility from the selected environment.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path


CHECKS = [
    ("numpy", True),
    ("quaternion", True),
    ("sophuspy", True),
    ("sophus", True),
    ("torch", True),
    ("torchvision", True),
    ("pytorch3d", True),
    ("torch_geometric", True),
    ("cv2", True),
    ("open3d", False),
    ("habitat_sim", True),
    ("habitat", True),
    ("habitat_baselines", True),
    ("home_robot", True),
    ("home_robot_sim", True),
]


def add_repo_paths(home_robot_root: Path, project_root: Path) -> None:
    candidates = [
        home_robot_root / "src" / "home_robot",
        home_robot_root / "src" / "home_robot_sim",
        home_robot_root / "src" / "third_party" / "habitat-lab" / "habitat-lab",
        home_robot_root / "src" / "third_party" / "habitat-lab" / "habitat-baselines",
        project_root / "shims",
    ]
    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path))
            print(f"PATH_ADDED {path}")
        else:
            print(f"PATH_MISSING {path}")


def check_import(module_name: str, required: bool, verbose_failures: bool) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - preflight should report all failures
        print(f"FAIL {module_name} {type(exc).__name__}: {str(exc)[:500]}")
        if verbose_failures:
            traceback.print_exc(limit=5)
        return not required

    version = getattr(module, "__version__", "no_version")
    location = getattr(module, "__file__", "no_file")
    print(f"OK {module_name} version={version} file={location}")
    return True


def check_torch_cuda() -> bool:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL torch_cuda_check {type(exc).__name__}: {str(exc)[:500]}")
        return False

    print(f"TORCH_VERSION {torch.__version__}")
    print(f"TORCH_CUDA_BUILT {getattr(torch.version, 'cuda', None)}")
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    print(f"TORCH_CUDA_AVAILABLE {available}")
    print(f"TORCH_CUDA_DEVICE_COUNT {count}")
    if available and count:
        for idx in range(count):
            print(f"TORCH_CUDA_DEVICE {idx} {torch.cuda.get_device_name(idx)}")
    return available and count > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--home-robot-root",
        default=os.environ.get("HOME_ROBOT_ROOT", "/root/workspace/tianshanzhang/benchmark/home-robot"),
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get("LMM_ROLLOUT_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])),
    )
    parser.add_argument("--verbose-failures", action="store_true")
    args = parser.parse_args()

    home_robot_root = Path(args.home_robot_root).resolve()
    project_root = Path(args.project_root).resolve()
    print(f"PYTHON_EXECUTABLE {sys.executable}")
    print(f"PYTHON_VERSION {sys.version.replace(os.linesep, ' ')}")
    print(f"CWD {Path.cwd()}")
    print(f"HOME_ROBOT_ROOT {home_robot_root}")
    print(f"LMM_ROLLOUT_PROJECT_ROOT {project_root}")

    add_repo_paths(home_robot_root, project_root)

    ok = True
    for module_name, required in CHECKS:
        ok = check_import(module_name, required, args.verbose_failures) and ok

    ok = check_torch_cuda() and ok
    print(f"IMPORT_PREFLIGHT_STATUS {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
