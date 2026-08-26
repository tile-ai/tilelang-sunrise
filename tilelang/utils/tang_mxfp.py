"""Helpers for S3 (stcuv2) sub-byte micro-scaling float formats (fp6 / fp4).

S3 tensor cores consume fp6 (e2m3 / e3m2) and fp4 (e2m1) operands as *uint8 byte
buffers* whose in-memory layout is format specific:

* ``fp8``  - 1 byte per element (low byte holds the e4m3 / e5m2 code).
* ``fp4``  - 1 byte per element in the ``mxf8f6f4`` *mixed* container
  (``eFP4_E2M1_MIX``): the 4-bit code lives in the low nibble so fp4 stays
  byte-aligned and can be mixed with fp6 / fp8 in the same row.  The pure
  ``mxf4`` / ``nvfp4`` kinds instead pack 2 codes per byte (handled by torch's
  ``float4_e2m1fn_x2`` dtype, not by this module).
* ``fp6``  - densely bit-packed in **16-element chunks**: every group of 16
  consecutive elements is packed LSB-first into 12 bytes and then zero-padded to
  16 bytes (a quarter of the 32-byte swizzle atom).  The padding makes every fp6
  row exactly ``K`` bytes wide, i.e. 1 byte per element on average, matching the
  fp8 / fp4 byte container.

Because there is no native torch / numpy dtype for fp6 (and fp4 in the mixed
container is awkward to build by hand), this module provides:

* :func:`quantize`  - round float values onto a format's representable grid and
  return integer codes.
* :func:`pack` / :func:`unpack`  - convert between integer codes and the S3 byte
  container layout.
* :func:`to_bytes` / :func:`from_bytes`  - one-shot float<->byte-container.
* :func:`ele_code`  - the tensor-core ``EleType`` value to pass to
  ``T.gemm(..., a_format=/b_format=)``.

All array I/O uses numpy; :func:`to_bytes` / :func:`from_bytes` also accept and
return torch tensors when torch is available.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FORMATS",
    "ele_code",
    "bits_of",
    "grid",
    "quantize",
    "pack",
    "unpack",
    "to_bytes",
    "from_bytes",
]

# name -> (EleType code, exponent bits, mantissa bits, bias).  Codes match
# tang::ptx::EleType (see cccl/tang/__ptx/instructions/tc_mma.h):
#   eFP8_E4M3=8 eFP8_E5M2=9 eFP6_E2M3=10 eFP6_E3M2=11 eFP4_E2M1_MIX=12
FORMATS = {
    "e4m3": (8, 4, 3, 7),
    "e5m2": (9, 5, 2, 15),
    "e2m3": (10, 2, 3, 1),
    "e3m2": (11, 3, 2, 3),
    "e2m1": (12, 2, 1, 1),  # fp4 in the mxf8f6f4 MIX container
}


def _spec(fmt: str):
    if fmt not in FORMATS:
        raise ValueError(f"unknown mxfp format {fmt!r}; choose from {list(FORMATS)}")
    return FORMATS[fmt]


def ele_code(fmt: str) -> int:
    """Tensor-core ``EleType`` value for ``T.gemm(a_format=/b_format=)``."""
    return _spec(fmt)[0]


def bits_of(fmt: str) -> int:
    """Number of significant bits of a code (1+exp+mant): fp8=8, fp6=6, fp4=4."""
    _, eb, mb, _ = _spec(fmt)
    return 1 + eb + mb


def grid(fmt: str) -> np.ndarray:
    """Decoded value for every integer code (length ``2**bits_of(fmt)``).

    The format has no inf / nan encodings; every bit pattern is a finite value.
    """
    _, eb, mb, bias = _spec(fmt)
    n = 1 << (1 + eb + mb)
    out = np.empty(n, dtype=np.float64)
    for v in range(n):
        s = (v >> (eb + mb)) & 1
        e = (v >> mb) & ((1 << eb) - 1)
        m = v & ((1 << mb) - 1)
        if e == 0:
            val = m / (1 << mb) * (2.0 ** (1 - bias))
        else:
            val = (1.0 + m / (1 << mb)) * (2.0 ** (e - bias))
        out[v] = -val if s else val
    return out


def quantize(values, fmt: str) -> np.ndarray:
    """Round real ``values`` onto ``fmt``'s grid, returning uint8 integer codes.

    Ties resolve to the nearest representable magnitude (round-to-nearest); the
    sign is preserved so ``-0`` maps to the negative-zero code when present.
    """
    g = grid(fmt)
    v = np.asarray(values, dtype=np.float64)
    # nearest code by absolute distance to every grid point
    idx = np.abs(v[..., None] - g[None, ...]).argmin(axis=-1)
    return idx.astype(np.uint8)


def _to_numpy(codes):
    try:
        import torch

        if isinstance(codes, torch.Tensor):
            return codes.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(codes)


_CHUNK = 16  # fp6 packs elements in 16-wide chunks (12 data + 4 pad bytes)


def pack(codes, fmt: str) -> np.ndarray:
    """Pack integer codes ``(..., K)`` into the S3 uint8 byte container ``(..., K)``.

    * fp8 / fp4: identity into the low byte / low nibble (1 byte per element).
    * fp6: 16-element chunks dense LSB-first into 12 bytes + 4 zero pad bytes.

    ``K`` must be a multiple of 16 for fp6.
    """
    c = _to_numpy(codes).astype(np.int64)
    bits = bits_of(fmt)
    K = c.shape[-1]
    flat = c.reshape(-1, K)
    if bits in (8,):
        return (flat & 0xFF).astype(np.uint8).reshape(c.shape)
    if bits == 4:
        return (flat & 0xF).astype(np.uint8).reshape(c.shape)
    # fp6: chunked dense pack
    if K % _CHUNK != 0:
        raise ValueError(f"fp6 K={K} must be a multiple of {_CHUNK}")
    out = np.zeros((flat.shape[0], K), dtype=np.uint8)
    for c0 in range(0, K, _CHUNK):
        chunk = flat[:, c0 : c0 + _CHUNK] & 0x3F  # (R,16)
        # accumulate 16 * 6 = 96 bits per row into 12 bytes (LSB-first)
        for i in range(_CHUNK):
            bitpos = i * bits
            for b in range(bits):
                src = (chunk[:, i] >> b) & 1
                byte = (bitpos + b) >> 3
                off = (bitpos + b) & 7
                out[:, c0 + byte] |= (src << off).astype(np.uint8)
        # bytes c0+12 .. c0+15 stay zero (pad)
    return out.reshape(c.shape[:-1] + (K,))


def unpack(buf, fmt: str) -> np.ndarray:
    """Inverse of :func:`pack`: uint8 container ``(..., K)`` -> codes ``(..., K)``."""
    b = _to_numpy(buf).astype(np.int64)
    bits = bits_of(fmt)
    K = b.shape[-1]
    flat = b.reshape(-1, K)
    if bits == 8:
        return (flat & 0xFF).astype(np.uint8).reshape(b.shape)
    if bits == 4:
        return (flat & 0xF).astype(np.uint8).reshape(b.shape)
    if K % _CHUNK != 0:
        raise ValueError(f"fp6 K={K} must be a multiple of {_CHUNK}")
    out = np.zeros((flat.shape[0], K), dtype=np.int64)
    for c0 in range(0, K, _CHUNK):
        for i in range(_CHUNK):
            bitpos = i * bits
            val = np.zeros(flat.shape[0], dtype=np.int64)
            for bb in range(bits):
                byte = (bitpos + bb) >> 3
                off = (bitpos + bb) & 7
                val |= ((flat[:, c0 + byte] >> off) & 1) << bb
            out[:, c0 + i] = val
    return out.astype(np.uint8).reshape(b.shape[:-1] + (K,))


def to_bytes(values, fmt: str):
    """Quantize real ``values`` and pack into the S3 byte container.

    Returns a torch ``uint8`` tensor if ``values`` is a torch tensor (or torch is
    importable and the input converts cleanly), otherwise a numpy ``uint8`` array.
    """
    arr = pack(quantize(values, fmt), fmt)
    try:
        import torch

        if isinstance(values, torch.Tensor):
            return torch.from_numpy(arr)
    except ImportError:
        pass
    return arr


def from_bytes(buf, fmt: str):
    """Decode the S3 byte container back to float64 values (numpy)."""
    return grid(fmt)[unpack(buf, fmt)]
