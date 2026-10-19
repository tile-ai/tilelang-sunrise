import os

import torch

IS_CUDA = torch.cuda.is_available()


def is_ptpu_available() -> bool:
    """Return whether the optional PyTorch PTPU backend is usable."""
    try:
        import torch_ptpu  # noqa: F401

        ptpu = getattr(torch, "ptpu", None)
        return ptpu is not None and bool(ptpu.is_available())
    except Exception:
        return False


IS_PTPU = is_ptpu_available()

IS_NPU = False
try:
    if not hasattr(torch, "npu"):
        import torch_npu  # noqa: F401
    IS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    IS_NPU = False
except Exception as e:
    print(f"An unexpected error occurred while checking NPU availability: {e}")

IS_MPS = False
try:
    IS_MPS = torch.backends.mps.is_available()
except AttributeError:
    print("MPS backend is not available in this PyTorch build.")
except Exception as e:
    print(f"An unexpected error occurred while checking MPS availability: {e}")


if IS_PTPU:
    Event = torch.ptpu.Event
elif IS_CUDA:
    Event = torch.cuda.Event
elif IS_NPU:
    Event = torch.npu.Event
elif IS_MPS:
    Event = torch.mps.Event
else:
    Event = None


def get_current_device():
    if is_ptpu_available():
        return torch.device("ptpu", torch.ptpu.current_device())
    if IS_CUDA:
        return torch.device("cuda", torch.cuda.current_device())
    if IS_NPU:
        return torch.device("npu", torch.npu.current_device())
    if IS_MPS:
        return torch.device("mps", 0)
    return torch.device("cpu")


def is_device_assert_supported() -> bool:
    """Whether any backend that lowers ``tl.device_assert`` is available.

    ``tl.device_assert`` / ``tl.device_assert_with_msg`` are lowered by both
    the CUDA and TANG codegens; on a host with neither backend the assert is
    meaningless and ``device_assert`` degrades to a no-op.
    """
    return IS_CUDA or is_ptpu_available()


def device_synchronize(device: int | torch.device | None = None) -> None:
    """Synchronize CUDA/HIP, MPS, or NPU, preferring CUDA when no device is given."""
    if device is None:
        if IS_PTPU:
            torch.ptpu.synchronize()
        elif IS_CUDA:
            torch.cuda.synchronize()
        elif IS_NPU:
            torch.npu.synchronize()
        elif IS_MPS:
            torch.mps.synchronize()
        else:
            raise RuntimeError("No device is available")
        return

    device_type = "cuda" if isinstance(device, int) else torch.device(device).type
    if device_type == "ptpu":
        torch.ptpu.synchronize(device)
    elif device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "npu":
        torch.npu.synchronize(device)
    elif device_type == "mps":
        torch.mps.synchronize()
    else:
        raise ValueError(f"device_synchronize only supports CUDA/HIP, MPS, or NPU devices, got {device}")


def get_available_cpu_count() -> int:
    """CPU cores available to this process (cpuset affinity), at least 1.

    Falls back to ``os.cpu_count`` where affinity is unavailable (macOS/Windows).
    """
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    return max(1, cpu_count or 1)
