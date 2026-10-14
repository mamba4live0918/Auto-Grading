"""Discover and activate the pinned Windows CUDA runtime packages."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType


PINNED_CUBLAS_VERSION = "12.9.2.10"
PINNED_CUDNN_VERSION = "9.24.0.43"
_PACKAGE_ORDER = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12")
_PINNED_PACKAGE_VERSIONS = {
    "nvidia-cublas-cu12": PINNED_CUBLAS_VERSION,
    "nvidia-cudnn-cu12": PINNED_CUDNN_VERSION,
}
_PACKAGE_RELATIVE_BINS = {
    "nvidia-cublas-cu12": Path("nvidia/cublas/bin"),
    "nvidia-cudnn-cu12": Path("nvidia/cudnn/bin"),
}
_REQUIRED_DLLS = {
    "nvidia-cublas-cu12": ("cublas64_12.dll", "cublasLt64_12.dll"),
    "nvidia-cudnn-cu12": (
        "cudnn64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_heuristic64_9.dll",
        "cudnn_ops64_9.dll",
    ),
}
_DLL_DIRECTORY_HANDLES: list[object] = []


class CudaRuntimeError(RuntimeError):
    """The pinned local Windows CUDA runtime is incomplete."""


@dataclass(frozen=True)
class CudaRuntimeStatus:
    """Non-secret runtime package and DLL completeness information."""

    ready: bool
    bin_directories: tuple[Path, ...]
    missing_dlls: tuple[str, ...]
    package_versions: Mapping[str, str]
    version_mismatches: Mapping[str, tuple[str, str]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "package_versions", MappingProxyType(dict(self.package_versions))
        )
        object.__setattr__(
            self, "version_mismatches", MappingProxyType(dict(self.version_mismatches))
        )


def inspect_cuda_runtime(
    *,
    package_bins: Mapping[str, Path] | None = None,
    package_versions: Mapping[str, str] | None = None,
) -> CudaRuntimeStatus:
    """Locate pinned package bins and verify the complete native DLL contract."""
    resolved_bins = (
        _installed_package_bins() if package_bins is None else dict(package_bins)
    )
    versions = (
        _installed_package_versions()
        if package_versions is None
        else dict(package_versions)
    )
    directories = tuple(
        Path(resolved_bins[name])
        for name in _PACKAGE_ORDER
        if name in resolved_bins
    )
    missing = tuple(
        filename
        for package in _PACKAGE_ORDER
        for filename in _REQUIRED_DLLS[package]
        if package not in resolved_bins
        or not (Path(resolved_bins[package]) / filename).is_file()
    )
    package_set_complete = all(name in resolved_bins and name in versions for name in _PACKAGE_ORDER)
    version_mismatches = {
        package: (_PINNED_PACKAGE_VERSIONS[package], versions[package])
        for package in _PACKAGE_ORDER
        if package in versions and versions[package] != _PINNED_PACKAGE_VERSIONS[package]
    }
    return CudaRuntimeStatus(
        ready=package_set_complete and not missing and not version_mismatches,
        bin_directories=directories,
        missing_dlls=missing,
        package_versions=versions,
        version_mismatches=version_mismatches,
    )


def activate_cuda_runtime(
    status: CudaRuntimeStatus | None = None,
    *,
    add_dll_directory: Callable[[str], object] | None = None,
) -> CudaRuntimeStatus:
    """Retain Windows DLL-directory handles and prepend native runtime bins."""
    actual = status or inspect_cuda_runtime()
    if not actual.ready:
        issues = ", ".join(actual.missing_dlls)
        if actual.version_mismatches:
            drifted = ", ".join(sorted(actual.version_mismatches))
            issues = f"{issues}; version mismatch: {drifted}" if issues else f"version mismatch: {drifted}"
        raise CudaRuntimeError(f"local CUDA runtime is incomplete: {issues or 'runtime packages'}")
    add_directory = add_dll_directory or getattr(os, "add_dll_directory", None)
    if add_directory is None:
        raise CudaRuntimeError("Windows DLL directory activation is unavailable")
    existing_path = os.environ.get("PATH", "")
    existing_entries = existing_path.split(os.pathsep) if existing_path else []
    new_entries: list[str] = []
    for directory in actual.bin_directories:
        directory_text = str(directory)
        _DLL_DIRECTORY_HANDLES.append(add_directory(directory_text))
        if directory_text not in existing_entries and directory_text not in new_entries:
            new_entries.append(directory_text)
    os.environ["PATH"] = os.pathsep.join([*new_entries, *existing_entries])
    return actual


def active_cuda_runtime_handle_count() -> int:
    """Return retained handle count for deterministic activation tests."""
    return len(_DLL_DIRECTORY_HANDLES)


def _installed_package_bins() -> dict[str, Path]:
    bins: dict[str, Path] = {}
    for package in _PACKAGE_ORDER:
        try:
            distribution = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            continue
        bins[package] = Path(
            distribution.locate_file(_PACKAGE_RELATIVE_BINS[package])
        )
    return bins


def _installed_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _PACKAGE_ORDER:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions
