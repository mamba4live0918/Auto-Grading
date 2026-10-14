"""Tests for deterministic Windows CUDA runtime discovery and activation."""

from __future__ import annotations

import os
from pathlib import Path

from presentation_assessment.research.cuda_runtime import (
    PINNED_CUBLAS_VERSION,
    PINNED_CUDNN_VERSION,
    activate_cuda_runtime,
    active_cuda_runtime_handle_count,
    inspect_cuda_runtime,
)


CUBLAS_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll")
CUDNN_DLLS = (
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
)


def _runtime_dirs(tmp_path: Path) -> dict[str, Path]:
    directories = {
        "nvidia-cublas-cu12": tmp_path / "cublas" / "bin",
        "nvidia-cudnn-cu12": tmp_path / "cudnn" / "bin",
    }
    for path in directories.values():
        path.mkdir(parents=True)
    for filename in CUBLAS_DLLS:
        (directories["nvidia-cublas-cu12"] / filename).write_bytes(b"runtime")
    for filename in CUDNN_DLLS:
        (directories["nvidia-cudnn-cu12"] / filename).write_bytes(b"runtime")
    return directories


def test_runtime_inspection_requires_complete_cublas_and_split_cudnn9_sets(
    tmp_path: Path,
) -> None:
    """Break caught: a lone cuDNN core DLL lets FULL_RUN reach a native failure."""
    directories = _runtime_dirs(tmp_path)
    complete = inspect_cuda_runtime(
        package_bins=directories,
        package_versions={
            "nvidia-cublas-cu12": "12.9.2.10",
            "nvidia-cudnn-cu12": "9.24.0.43",
        },
    )

    assert complete.ready is True
    assert complete.missing_dlls == ()
    assert complete.package_versions == {
        "nvidia-cublas-cu12": "12.9.2.10",
        "nvidia-cudnn-cu12": "9.24.0.43",
    }

    (directories["nvidia-cudnn-cu12"] / "cudnn_graph64_9.dll").unlink()
    incomplete = inspect_cuda_runtime(
        package_bins=directories,
        package_versions=complete.package_versions,
    )

    assert incomplete.ready is False
    assert incomplete.missing_dlls == ("cudnn_graph64_9.dll",)


def test_runtime_inspection_rejects_versions_outside_the_locked_binding(
    tmp_path: Path,
) -> None:
    """Break caught: complete DLLs from a drifted package version pass preflight."""
    directories = _runtime_dirs(tmp_path)

    wrong_cublas = inspect_cuda_runtime(
        package_bins=directories,
        package_versions={
            "nvidia-cublas-cu12": "12.9.2.11",
            "nvidia-cudnn-cu12": PINNED_CUDNN_VERSION,
        },
    )
    wrong_cudnn = inspect_cuda_runtime(
        package_bins=directories,
        package_versions={
            "nvidia-cublas-cu12": PINNED_CUBLAS_VERSION,
            "nvidia-cudnn-cu12": "9.24.0.44",
        },
    )

    assert wrong_cublas.ready is False
    assert wrong_cublas.version_mismatches == {
        "nvidia-cublas-cu12": (PINNED_CUBLAS_VERSION, "12.9.2.11")
    }
    assert wrong_cudnn.ready is False
    assert wrong_cudnn.version_mismatches == {
        "nvidia-cudnn-cu12": (PINNED_CUDNN_VERSION, "9.24.0.44")
    }


def test_runtime_activation_retains_directory_handles_and_prepends_path(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Break caught: DLL directories disappear before CTranslate2 resolves symbols."""
    directories = _runtime_dirs(tmp_path)
    status = inspect_cuda_runtime(
        package_bins=directories,
        package_versions={
            "nvidia-cublas-cu12": "12.9.2.10",
            "nvidia-cudnn-cu12": "9.24.0.43",
        },
    )
    handles: list[object] = []

    def add_directory(path: str) -> object:
        handle = object()
        handles.append(handle)
        return handle

    monkeypatch.setenv("PATH", "existing")
    before = active_cuda_runtime_handle_count()

    activated = activate_cuda_runtime(status, add_dll_directory=add_directory)

    assert activated.ready is True
    assert active_cuda_runtime_handle_count() == before + 2
    assert len(handles) == 2
    path_entries = os.environ["PATH"].split(os.pathsep)
    assert path_entries[:2] == [
        str(directories["nvidia-cublas-cu12"]),
        str(directories["nvidia-cudnn-cu12"]),
    ]
