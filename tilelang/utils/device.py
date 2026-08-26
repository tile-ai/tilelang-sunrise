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

IS_MPS = False
try:
    IS_MPS = torch.backends.mps.is_available()
except AttributeError:
    print("MPS backend is not available in this PyTorch build.")
except Exception as e:
    print(f"An unexpected error occurred while checking MPS availability: {e}")


def get_current_device():
    if is_ptpu_available():
        return torch.device("ptpu", torch.ptpu.current_device())
    if IS_CUDA:
        return torch.device("cuda", torch.cuda.current_device())
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
