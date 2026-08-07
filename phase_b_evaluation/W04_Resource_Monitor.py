"""Reusable latency and resource instrumentation for Week 4 evaluations.

The module is intentionally independent of a candidate model. Text, RAG, and
VLM runners can wrap their request stages with :class:`RequestProfiler` and
join the resulting dictionary to the row-level quality evidence by request ID.

Unavailable measurements are represented by ``None`` plus an availability
reason. They are never coerced to zero.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Sequence


MIB = 1024 * 1024


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _mib(value_bytes: int | float | None) -> float | None:
    if value_bytes is None:
        return None
    return round(float(value_bytes) / MIB, 6)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Return a linearly interpolated percentile without NumPy."""

    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1")
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def summarize_numeric(values: Sequence[float | int | None]) -> dict[str, Any]:
    """Create the required descriptive latency/resource summary."""

    clean = [float(value) for value in values if value is not None]
    clean = [value for value in clean if math.isfinite(value)]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "stdev": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "mean": round(statistics.fmean(clean), 6),
        "stdev": round(statistics.stdev(clean), 6) if len(clean) > 1 else 0.0,
        "p50": round(percentile(clean, 0.50) or 0.0, 6),
        "p90": round(percentile(clean, 0.90) or 0.0, 6),
        "p95": round(percentile(clean, 0.95) or 0.0, 6),
        "max": round(max(clean), 6),
    }


def directory_size_bytes(path: str | Path) -> int | None:
    """Return an approximate checkpoint size without following symlinks."""

    root = Path(path)
    if not root.exists():
        return None
    if root.is_file():
        return root.stat().st_size
    total = 0
    for candidate in root.rglob("*"):
        try:
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def _optional_import(name: str) -> tuple[Any | None, str | None]:
    try:
        module = __import__(name)
        return module, None
    except Exception as exc:  # pragma: no cover - environment-dependent
        return None, f"{type(exc).__name__}: {exc}"


def _query_nvidia_smi(
    executable: str = "nvidia-smi", gpu_index: int = 0
) -> tuple[dict[str, Any] | None, str | None]:
    fields = [
        "index",
        "name",
        "memory.total",
        "memory.used",
        "utilization.gpu",
        "power.draw",
    ]
    command = [
        executable,
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"

    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != len(fields):
            continue
        try:
            parsed.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": float(parts[2]),
                    "memory_used_mib": float(parts[3]),
                    "utilization_pct": float(parts[4]),
                    "power_w": None
                    if parts[5].lower() in {"n/a", "[n/a]", "not supported"}
                    else float(parts[5]),
                }
            )
        except ValueError:
            continue
    for row in parsed:
        if row["index"] == gpu_index:
            return row, None
    if parsed:
        return None, f"gpu index {gpu_index} not present in nvidia-smi output"
    return None, "nvidia-smi returned no parseable GPU rows"


class ResourceSampler:
    """Low-frequency host/GPU sampler plus PyTorch peak counters."""

    def __init__(
        self,
        sample_interval_s: float = 0.25,
        nvidia_smi_interval_s: float = 1.0,
        gpu_index: int = 0,
        nvidia_smi_executable: str = "nvidia-smi",
        enable_torch_probe: bool = True,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be positive")
        if nvidia_smi_interval_s <= 0:
            raise ValueError("nvidia_smi_interval_s must be positive")
        self.sample_interval_s = sample_interval_s
        self.nvidia_smi_interval_s = nvidia_smi_interval_s
        self.gpu_index = gpu_index
        self.nvidia_smi_executable = nvidia_smi_executable
        self.enable_torch_probe = enable_torch_probe
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._host_samples: list[dict[str, float]] = []
        self._gpu_samples: list[dict[str, Any]] = []
        self._availability: dict[str, Any] = {}
        self._torch_before: dict[str, float | None] = {}
        self._torch_after: dict[str, float | None] = {}
        self.started_at_utc: str | None = None
        self.ended_at_utc: str | None = None

    def _sample_host(self) -> None:
        psutil, reason = _optional_import("psutil")
        if psutil is None:
            self._availability["host"] = {"available": False, "reason": reason}
            return
        try:
            process = psutil.Process(os.getpid())
            virtual = psutil.virtual_memory()
            sample = {
                "monotonic_ns": float(time.perf_counter_ns()),
                "process_rss_mib": _mib(process.memory_info().rss) or 0.0,
                "system_used_mib": _mib(virtual.used) or 0.0,
                "system_total_mib": _mib(virtual.total) or 0.0,
            }
            with self._lock:
                self._host_samples.append(sample)
            self._availability["host"] = {"available": True, "reason": None}
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._availability["host"] = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def _sample_gpu(self) -> None:
        sample, reason = _query_nvidia_smi(
            self.nvidia_smi_executable, self.gpu_index
        )
        if sample is None:
            self._availability["nvidia_smi"] = {
                "available": False,
                "reason": reason,
            }
            return
        sample["monotonic_ns"] = float(time.perf_counter_ns())
        with self._lock:
            self._gpu_samples.append(sample)
        self._availability["nvidia_smi"] = {"available": True, "reason": None}

    def _torch_snapshot(self, reset_peak: bool) -> dict[str, float | None]:
        if not self.enable_torch_probe:
            self._availability["torch_cuda"] = {
                "available": False,
                "reason": "torch probe disabled",
            }
            return {
                "allocated_mib": None,
                "reserved_mib": None,
                "peak_allocated_mib": None,
                "peak_reserved_mib": None,
            }
        torch, reason = _optional_import("torch")
        if torch is None:
            self._availability["torch_cuda"] = {
                "available": False,
                "reason": reason,
            }
            return {
                "allocated_mib": None,
                "reserved_mib": None,
                "peak_allocated_mib": None,
                "peak_reserved_mib": None,
            }
        try:
            if not torch.cuda.is_available():
                self._availability["torch_cuda"] = {
                    "available": False,
                    "reason": "torch.cuda.is_available() is False",
                }
                return {
                    "allocated_mib": None,
                    "reserved_mib": None,
                    "peak_allocated_mib": None,
                    "peak_reserved_mib": None,
                }
            device = torch.device(f"cuda:{self.gpu_index}")
            if reset_peak:
                torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            self._availability["torch_cuda"] = {"available": True, "reason": None}
            return {
                "allocated_mib": _mib(torch.cuda.memory_allocated(device)),
                "reserved_mib": _mib(torch.cuda.memory_reserved(device)),
                "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved(device)),
            }
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._availability["torch_cuda"] = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            return {
                "allocated_mib": None,
                "reserved_mib": None,
                "peak_allocated_mib": None,
                "peak_reserved_mib": None,
            }

    def _run(self) -> None:
        next_gpu = 0.0
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            self._sample_host()
            if loop_start >= next_gpu:
                self._sample_gpu()
                next_gpu = loop_start + self.nvidia_smi_interval_s
            elapsed = time.perf_counter() - loop_start
            self._stop.wait(max(0.0, self.sample_interval_s - elapsed))

    def start(self) -> "ResourceSampler":
        if self._thread is not None:
            raise RuntimeError("ResourceSampler has already been started")
        self.started_at_utc = _utc_now()
        self._torch_before = self._torch_snapshot(reset_peak=True)
        self._sample_host()
        self._sample_gpu()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        if self._thread is None:
            raise RuntimeError("ResourceSampler has not been started")
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.sample_interval_s * 4))
        self._sample_host()
        self._sample_gpu()
        self._torch_after = self._torch_snapshot(reset_peak=False)
        self.ended_at_utc = _utc_now()
        self._thread = None
        return self.summary()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            host = list(self._host_samples)
            gpu = list(self._gpu_samples)

        def first_peak_after(
            rows: Sequence[dict[str, Any]], key: str
        ) -> dict[str, float | None]:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            if not values:
                return {"before": None, "peak": None, "after": None}
            return {
                "before": round(values[0], 6),
                "peak": round(max(values), 6),
                "after": round(values[-1], 6),
            }

        utilization = [
            float(row["utilization_pct"])
            for row in gpu
            if row.get("utilization_pct") is not None
        ]
        power = [
            float(row["power_w"])
            for row in gpu
            if row.get("power_w") is not None
        ]
        total_host = host[-1]["system_total_mib"] if host else None
        total_gpu = gpu[-1]["memory_total_mib"] if gpu else None
        gpu_name = gpu[-1]["name"] if gpu else None
        return {
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "sample_interval_ms": round(self.sample_interval_s * 1000, 3),
            "nvidia_smi_interval_ms": round(
                self.nvidia_smi_interval_s * 1000, 3
            ),
            "host_sample_count": len(host),
            "gpu_sample_count": len(gpu),
            "availability": dict(self._availability),
            "process_rss_mib": first_peak_after(host, "process_rss_mib"),
            "system_memory_used_mib": first_peak_after(host, "system_used_mib"),
            "system_memory_total_mib": total_host,
            "gpu_name": gpu_name,
            "gpu_device_memory_used_mib": first_peak_after(gpu, "memory_used_mib"),
            "gpu_device_memory_total_mib": total_gpu,
            "gpu_utilization_pct": {
                "mean": round(statistics.fmean(utilization), 6)
                if utilization
                else None,
                "p95": round(percentile(utilization, 0.95) or 0.0, 6)
                if utilization
                else None,
                "peak": round(max(utilization), 6) if utilization else None,
            },
            "gpu_power_w": {
                "mean": round(statistics.fmean(power), 6) if power else None,
                "peak": round(max(power), 6) if power else None,
            },
            "torch_cuda": {
                "allocated_before_mib": self._torch_before.get("allocated_mib"),
                "allocated_after_mib": self._torch_after.get("allocated_mib"),
                "allocated_peak_mib": self._torch_after.get("peak_allocated_mib"),
                "reserved_before_mib": self._torch_before.get("reserved_mib"),
                "reserved_after_mib": self._torch_after.get("reserved_mib"),
                "reserved_peak_mib": self._torch_after.get("peak_reserved_mib"),
            },
        }


class RequestProfiler:
    """Context manager for request stages, TTFT, and resource evidence."""

    def __init__(
        self,
        request_id: str,
        *,
        sample_interval_s: float = 0.25,
        nvidia_smi_interval_s: float = 1.0,
        gpu_index: int = 0,
        nvidia_smi_executable: str = "nvidia-smi",
        enable_torch_probe: bool = True,
    ) -> None:
        if not request_id:
            raise ValueError("request_id is required")
        self.request_id = request_id
        self.sampler = ResourceSampler(
            sample_interval_s=sample_interval_s,
            nvidia_smi_interval_s=nvidia_smi_interval_s,
            gpu_index=gpu_index,
            nvidia_smi_executable=nvidia_smi_executable,
            enable_torch_probe=enable_torch_probe,
        )
        self._request_start_ns: int | None = None
        self._request_end_ns: int | None = None
        self._first_token_ns: int | None = None
        self._stage_records: list[dict[str, Any]] = []
        self._resource_summary: dict[str, Any] | None = None
        self._error: dict[str, str] | None = None

    def __enter__(self) -> "RequestProfiler":
        # Establish the resource baseline before starting the user-visible
        # request clock.  In particular, the initial nvidia-smi subprocess must
        # not be charged to question-to-response latency.
        self.sampler.start()
        self._request_start_ns = time.perf_counter_ns()
        return self

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not name or not name.endswith("_ms"):
            raise ValueError("stage name must be non-empty and end in '_ms'")
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            ended_ns = time.perf_counter_ns()
            self._stage_records.append(
                {
                    "name": name,
                    "started_offset_ms": self._offset_ms(started_ns),
                    "duration_ms": round((ended_ns - started_ns) / 1_000_000, 6),
                }
            )

    def mark_first_token(self) -> None:
        if self._request_start_ns is None:
            raise RuntimeError("profiler has not been entered")
        if self._first_token_ns is None:
            self._first_token_ns = time.perf_counter_ns()

    def _offset_ms(self, timestamp_ns: int) -> float:
        if self._request_start_ns is None:
            raise RuntimeError("profiler has not been entered")
        return round((timestamp_ns - self._request_start_ns) / 1_000_000, 6)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc is not None:
            self._error = {"type": type(exc).__name__, "message": str(exc)}
        # Stop the user-visible request clock before sampler teardown for the
        # same reason that sampler setup is excluded in __enter__.
        self._request_end_ns = time.perf_counter_ns()
        self._resource_summary = self.sampler.stop()
        return False

    def result(self) -> dict[str, Any]:
        if self._request_start_ns is None or self._request_end_ns is None:
            raise RuntimeError("result is available only after the context exits")
        totals: dict[str, float] = {}
        for record in self._stage_records:
            totals[record["name"]] = round(
                totals.get(record["name"], 0.0) + record["duration_ms"], 6
            )
        end_to_end_ms = round(
            (self._request_end_ns - self._request_start_ns) / 1_000_000, 6
        )
        ttft_ms = (
            round((self._first_token_ns - self._request_start_ns) / 1_000_000, 6)
            if self._first_token_ns is not None
            else None
        )
        if any(duration < 0 for duration in totals.values()):
            raise AssertionError("negative stage duration")
        if any(duration > end_to_end_ms + 1.0 for duration in totals.values()):
            raise AssertionError("child stage exceeds end-to-end duration")
        return {
            "request_id": self.request_id,
            "timings": {
                **totals,
                "ttft_ms": ttft_ms,
                "question_to_response_ms": end_to_end_ms,
                "stage_records": list(self._stage_records),
            },
            "measurement_availability": {
                "ttft_ms": {
                    "available": ttft_ms is not None,
                    "reason": None
                    if ttft_ms is not None
                    else "no generated token was observed by the streamer",
                }
            },
            "resources": self._resource_summary,
            "error": self._error,
        }


def environment_manifest(
    *,
    model_path: str | Path | None = None,
    gpu_index: int = 0,
    import_heavy_modules: bool = True,
) -> dict[str, Any]:
    """Collect static environment evidence once per model run."""

    psutil, psutil_reason = _optional_import("psutil")
    if import_heavy_modules:
        torch, torch_reason = _optional_import("torch")
        transformers, transformers_reason = _optional_import("transformers")
        torch_version = getattr(torch, "__version__", None)
        transformers_version = getattr(transformers, "__version__", None)
        cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
    else:
        torch = None
        transformers = None
        torch_reason = "heavy module import disabled"
        transformers_reason = "heavy module import disabled"
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = None
        try:
            transformers_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            transformers_version = None
        cuda_runtime = None
    gpu, gpu_reason = _query_nvidia_smi(gpu_index=gpu_index)
    manifest: dict[str, Any] = {
        "captured_at_utc": _utc_now(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "host_ram_total_mib": None,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "cuda_runtime": cuda_runtime,
        "gpu": gpu,
        "checkpoint_path": str(model_path) if model_path is not None else None,
        "checkpoint_bytes": directory_size_bytes(model_path)
        if model_path is not None
        else None,
        "availability": {
            "psutil": {"available": psutil is not None, "reason": psutil_reason},
            "torch": {
                "available": torch is not None if import_heavy_modules else torch_version is not None,
                "reason": torch_reason,
            },
            "transformers": {
                "available": transformers is not None
                if import_heavy_modules
                else transformers_version is not None,
                "reason": transformers_reason,
            },
            "nvidia_smi": {"available": gpu is not None, "reason": gpu_reason},
        },
    }
    if psutil is not None:
        try:
            manifest["host_ram_total_mib"] = _mib(psutil.virtual_memory().total)
        except Exception as exc:  # pragma: no cover - environment-dependent
            manifest["availability"]["psutil"] = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
    return manifest


def validate_jsonable(payload: dict[str, Any]) -> None:
    """Raise TypeError early if a trace contains non-serializable objects."""

    json.dumps(payload, ensure_ascii=False, allow_nan=False)


__all__ = [
    "RequestProfiler",
    "ResourceSampler",
    "directory_size_bytes",
    "environment_manifest",
    "percentile",
    "summarize_numeric",
    "validate_jsonable",
]
