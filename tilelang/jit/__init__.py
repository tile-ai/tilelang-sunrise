"""
This module provides an auto-tuning infrastructure for TileLang (tl) programs.
It includes functionality to JIT-compile TileLang programs into a runnable
kernel adapter using TVM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import (
    Any,
    Generic,
    TypeVar,
    overload,
    Literal,
    ParamSpec,
)
from collections.abc import Callable
from collections.abc import Iterable

from tilelang import tvm as tvm
from tilelang.language.eager import PrimFunc, prim_func, JITFunc
from tvm.target import Target

from tilelang.jit.kernel import JITKernel
from tilelang.cache import cached
from os import path, makedirs
from logging import getLogger
from tilelang.jit.param import Kernel
from tilelang.transform.pass_config import depythonize_pass_config_value
import concurrent.futures

from tqdm.auto import tqdm

logger = getLogger(__name__)

_P = ParamSpec("_P")
_KP = ParamSpec("_KP")
_T = TypeVar("_T")
_Ret = TypeVar("_Ret")
TargetLike = str | dict[str, object] | Target
_OuterConfigKey = tuple[tuple[str, Any], ...]
_CallFormKey = tuple[tuple[Any, ...], tuple[tuple[str, Any], ...], _OuterConfigKey]
_CALL_FORM_CACHE_MISS = object()
# Marks a _kernel_cache entry with no recorded PassContext, i.e. one injected
# directly rather than compiled through _store_kernel_cache. Such an entry is
# served under any context.
_KERNEL_CACHE_ANY_CONTEXT = object()


def _outer_pass_config_key() -> _OuterConfigKey:
    """Hashable snapshot of the ``tl.*`` / ``tirx.*`` entries of the enclosing
    PassContext.

    This must be part of the call-form cache key. The cache memoizes kernels by
    raw Python call form, but the same call form compiled under two different
    PassContexts is two different kernels: without this, the first call wins and
    every later ``with PassContext(config=...)`` around an identical call is
    silently ignored, returning a kernel built with the wrong config.

    Values are keyed through ``depythonize_pass_config_value`` rather than a
    bare ``str(v)`` so that the cache key is derived from the same
    normalization the compiled config goes through (see the merge below and in
    ``JITKernel._compile_and_create_adapter``). ``str()`` on a raw FFI value
    yields whatever the TIR printer happens to emit -- ``T.bool(True)`` for a
    bool, ``("-O3",)`` for an ``ffi.Array`` -- so distinctness would rest on
    printer formatting, which is not a stable contract across TVM versions.
    ``repr()`` on the depythonized value keeps ``1``/``True``/``'1'`` apart on
    plain Python types instead.
    """
    try:
        ctx = tvm.transform.PassContext.current()
        config = getattr(ctx, "config", None)
        if not config:
            return ()
        # NOTE: depythonize recurses into Mappings, and repr() of a dict is
        # insertion-ordered -- two equal-but-differently-ordered dict configs
        # would key differently. No tl.* config is dict-valued today; sort the
        # items here if one is ever added.
        return tuple(
            sorted(
                (k, repr(depythonize_pass_config_value(v)))
                for k, v in config.items()
                if isinstance(k, str) and (k.startswith("tl.") or k.startswith("tirx."))
            )
        )
    except (AttributeError, TypeError):
        return ()


@dataclass
class _CallFormCache:
    """Memoize lazy no-tensor kernel factories by raw Python call form."""

    entries: dict[_CallFormKey, Kernel] = field(default_factory=dict)
    last_args: tuple[Any, ...] | None = None
    last_kwargs: dict[str, Any] | None = None
    last_outer: _OuterConfigKey | None = None
    last_kernel: Kernel | object = _CALL_FORM_CACHE_MISS

    def clear(self) -> None:
        self.entries.clear()
        self.last_args = None
        self.last_kwargs = None
        self.last_outer = None
        self.last_kernel = _CALL_FORM_CACHE_MISS

    def __len__(self) -> int:
        return len(self.entries)

    def _matches_last(self, args: tuple[Any, ...], kwargs: dict[str, Any], outer: _OuterConfigKey) -> bool:
        if self.last_args is None or self.last_kwargs is None:
            return False
        # outer must be compared too, or the fast path hands back a kernel
        # compiled under a different PassContext.
        return args == self.last_args and kwargs == self.last_kwargs and outer == self.last_outer

    def _remember(self, call_form_key: _CallFormKey, kernel: Kernel) -> None:
        args, kwargs_items, outer = call_form_key
        self.last_args = args
        self.last_kwargs = dict(kwargs_items)
        self.last_outer = outer
        self.last_kernel = kernel

    def lookup(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Kernel | object, _CallFormKey | None]:
        # The enclosing PassContext is part of the identity of the compiled
        # kernel, so it belongs in the key (see _outer_pass_config_key).
        outer = _outer_pass_config_key()

        # Fastest path for tight loops: avoid rebuilding and hashing the call-form key.
        if self._matches_last(args, kwargs, outer):
            return self.last_kernel, None

        call_form_key = (args, tuple(kwargs.items()), outer)
        kernel = self.entries.get(call_form_key, _CALL_FORM_CACHE_MISS)
        if kernel is not _CALL_FORM_CACHE_MISS:
            self._remember(call_form_key, kernel)
        return kernel, call_form_key

    def store(self, call_form_key: _CallFormKey, kernel: Kernel) -> None:
        self.entries[call_form_key] = kernel
        self._remember(call_form_key, kernel)


def compile(
    func: PrimFunc[_KP, _T] = None,
    out_idx: list[int] | int | None = None,
    execution_backend: Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "simulator"] | None = None,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel[_KP, _T]:
    """
    Compile the given TileLang PrimFunc with TVM and build a JITKernel.

    Parameters
    ----------
    func : tvm.tirx.PrimFunc, optional
        The TileLang TIR function to compile and wrap.
    out_idx : Union[List[int], int], optional
        Index(es) of the output tensors to return (default: None).
    execution_backend : Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "simulator"], optional
        Execution backend to use for kernel execution. If None, reads from
        TILELANG_EXECUTION_BACKEND environment variable (defaults to "auto").
    target : str, dict, or tvm.target.Target, optional
        Compilation target. If None, reads from TILELANG_DEFAULT_TARGET environment
        variable (defaults to "auto"). Use a dict for target attributes, for example
        {"kind": "cuda", "arch": "sm_90"}.
    target_host : str, dict, or tvm.target.Target, optional
        Target host for cross-compilation (default: None).
    verbose : bool, optional
        Whether to enable verbose output. If None, reads from
        TILELANG_VERBOSE environment variable (defaults to False).
    pass_configs : dict, optional
        Additional keyword arguments to pass to the Compiler PassContext.
        Refer to `tilelang.transform.PassConfigKey` for supported options.

    Environment Variables
    ---------------------
    TILELANG_DEFAULT_TARGET : str
        Default compilation target (e.g., "cuda", "llvm", or a JSON target config string).
        Defaults to "auto".
    TILELANG_EXECUTION_BACKEND : str
        Default execution backend. Defaults to "auto".
    TILELANG_VERBOSE : str
        Set to "1", "true", "yes", or "on" to enable verbose compilation by default.
    """

    assert isinstance(func, PrimFunc), f"target function must be a PrimFunc but got {type(func)}"

    # Merge function-level attrs from PrimFunc
    func_attrs = func.attrs
    if func_attrs and "tilelang_out_idx" in func_attrs:
        func_out_idx = list(func_attrs["tilelang_out_idx"])
        if out_idx is not None:
            raise ValueError("Out index conflict: out_idx is specified and prim_func have returned `T.empty` tensors")
        out_idx = func_out_idx
    if func_attrs and "tilelang_pass_configs" in func_attrs:
        func_pc = {str(k): depythonize_pass_config_value(v) for k, v in func_attrs["tilelang_pass_configs"].items()}
        if pass_configs is not None:
            # External pass_configs override function-level ones
            func_pc.update(pass_configs)
        pass_configs = func_pc

    # Merge configs from the current (outer) PassContext so that user-set
    # pass configs (e.g. TL_USE_ASYNC_COP4) are inherited. Only keys with a
    # ``tl.`` or ``tirx.`` prefix are inherited — arbitrary PassContext
    # entries are not TileLang pass configs and must not leak in.
    #
    # This merge MUST happen before the cache-key computation in cached() so
    # that two calls with different outer PassContext settings produce
    # different cache keys and do not silently return a kernel compiled with
    # the wrong config.
    #
    # Normalize None to {} *before* the merge: `k not in None` raises TypeError,
    # which the except below would swallow as a warning, silently dropping the
    # whole outer context for every kernel that passes no explicit pass_configs
    # (the common case).
    #
    # Copy rather than mutate: for a @tilelang.jit kernel this dict is the
    # decorator's own self.pass_configs, so merging into it in place would make
    # every inherited key permanently part of the kernel's config and leak
    # across subsequent calls under different PassContexts.
    #
    # Depythonize the inherited values: the outer context's config has been
    # through the FFI, so a Python ``True`` reads back as ``IntImm`` and a list
    # as ``ffi.Array``. Storing those would make the kernel-cache key's
    # json.dumps raise TypeError (visible only when the cache is enabled).
    pass_configs = {} if pass_configs is None else dict(pass_configs)
    try:
        outer_ctx = tvm.transform.PassContext.current()
        if outer_ctx is not None and hasattr(outer_ctx, "config") and outer_ctx.config is not None:
            for k, v in outer_ctx.config.items():
                if isinstance(k, str) and (k.startswith("tl.") or k.startswith("tirx.")) and k not in pass_configs:
                    pass_configs[k] = depythonize_pass_config_value(v)
    except (AttributeError, TypeError) as e:
        logger.warning("Failed to merge outer PassContext configs: %s", e)

    if func_attrs and "tilelang_compile_flags" in func_attrs:
        func_cf = list(func_attrs["tilelang_compile_flags"])
        if compile_flags is not None:
            if isinstance(compile_flags, str):
                func_cf.append(compile_flags)
            else:
                func_cf.extend(compile_flags)
        compile_flags = func_cf

    return cached(
        func=func,
        out_idx=out_idx,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )


def par_compile(
    funcs: Iterable[PrimFunc[_KP, _T]],
    out_idx: list[int] | int | None = None,
    execution_backend: Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "simulator"] | None = None,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
    num_workers: int | None = None,
    ignore_error: bool = False,
) -> list[JITKernel[_KP, _T]]:
    """
    Parallel compile multiple TileLang PrimFunc with TVM and build JITKernels.

    Parameters
    ----------
    funcs : Iterable[tvm.tirx.PrimFunc]
        The TileLang TIR functions to compile and wrap.
    out_idx : Union[List[int], int], optional
        Index(es) of the output tensors to return (default: None).
    execution_backend : Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "simulator"], optional
        Execution backend to use for kernel execution. If None, reads from
        TILELANG_EXECUTION_BACKEND environment variable (defaults to "auto").
    target : str, dict, or tvm.target.Target, optional
        Compilation target. If None, reads from TILELANG_DEFAULT_TARGET environment
        variable (defaults to "auto"). Use a dict for target attributes, for example
        {"kind": "cuda", "arch": "sm_90"}.
    target_host : str, dict, or tvm.target.Target, optional
        Target host for cross-compilation (default: None).
    verbose : bool, optional
        Whether to enable verbose output. If None, reads from
        TILELANG_VERBOSE environment variable (defaults to False).
    pass_configs : dict, optional
        Additional keyword arguments to pass to the Compiler PassContext.
        Refer to `tilelang.transform.PassConfigKey` for supported options.

    Environment Variables
    ---------------------
    TILELANG_DEFAULT_TARGET : str
        Default compilation target (e.g., "cuda", "llvm", or a JSON target config string).
        Defaults to "auto".
    TILELANG_EXECUTION_BACKEND : str
        Default execution backend. Defaults to "auto".
    TILELANG_VERBOSE : str
        Set to "1", "true", "yes", or "on" to enable verbose compilation by default.
    """

    with concurrent.futures.ThreadPoolExecutor(num_workers, "tl-par-comp") as executor:
        futures = []
        future_map = {}
        for i, func in enumerate(funcs):
            future = executor.submit(
                compile,
                func=func,
                out_idx=out_idx,
                execution_backend=execution_backend,
                target=target,
                target_host=target_host,
                verbose=verbose,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
            future_map[future] = i
            futures.append(future)
        results = [... for _ in futures]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Parallel Compiling",
        ):
            idx = future_map[future]
            if ignore_error:
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Error compiling function at index {idx}: {e}")
                    results[idx] = None
            else:
                results[idx] = future.result()
        return results
    return results


class _PaddedKernel:
    """Wraps a compiled GEMM kernel, padding A/B inputs and slicing the output.

    Data-layer padding for ``tl.gemm_pad_m/n/k`` (extra padding size).  The
    wrapped kernel was compiled for the padded shapes (M+pad_m, N+pad_n,
    K+pad_k), so on invocation we zero-pad A (M=dim0, K=dim1) and B (K=dim0,
    N=dim1) up to those shapes, run, and slice the result back to (M, N).
    """

    def __init__(self, kernel, pad_m, pad_n, pad_k, orig_m, orig_n, cache_size=4):
        self._kernel = kernel
        self._pad_m, self._pad_n, self._pad_k = pad_m, pad_n, pad_k
        self._orig_m, self._orig_n = orig_m, orig_n
        # Bounded LRU cache of padded A/B, keyed by object identity and holding
        # the input references (so their id() cannot be reused).  Lets a few
        # distinct input pairs be invoked alternately without re-padding; the
        # bound avoids unbounded accumulation for streaming inputs.
        self._cache_size = cache_size
        self._cache = {}  # (id(A), id(B)) -> (A, B, A_p, B_p)
        self._lru = []

    def __call__(self, *args, **kwargs):
        import torch

        A, B = args[0], args[1]
        key = (id(A), id(B))
        entry = self._cache.get(key)
        if entry is not None:
            _, _, A_p, B_p = entry
            self._lru.remove(key)
            self._lru.append(key)
        else:
            # A: pad M (dim0) and K (dim1).  B: pad K (dim0) and N (dim1).
            A_p = torch.nn.functional.pad(A, (0, self._pad_k, 0, self._pad_m))
            B_p = torch.nn.functional.pad(B, (0, self._pad_n, 0, self._pad_k))
            self._cache[key] = (A, B, A_p, B_p)
            self._lru.append(key)
            if len(self._lru) > self._cache_size:
                self._cache.pop(self._lru.pop(0), None)
        C = self._kernel(A_p, B_p, *args[2:], **kwargs)
        return C[: self._orig_m, : self._orig_n]


@dataclass
class JITImpl(Generic[_P, _KP, _T, _Ret]):
    """
    Just-In-Time compilation wrapper for TileLang programs.

    This class provides a unified interface for compiling and executing TileLang
    kernels. It supports two execution modes that are automatically inferred:

    Execution Modes
    ---------------
    - **lazy**: The decorated function returns a PrimFunc explicitly. Calling the
      JIT wrapper returns a compiled kernel object, which can be invoked separately.
      This mode is useful when you want to inspect or reuse the kernel object.

      Example (lazy mode)::

          @tilelang.jit(out_idx=[-1])
          def matmul(M, N, K, block_M, block_N, block_K):
              @T.prim_func
              def kernel(A: T.Tensor((M, K), dtype), ...):
                  ...
              return kernel  # explicitly return PrimFunc

          kernel = matmul(1024, 1024, 1024, 128, 128, 32)  # returns kernel
          result = kernel(a, b)  # execute separately

    - **eager**: The decorated function uses the DSL builder pattern with tensor
      type annotations. Calling the JIT wrapper compiles and immediately executes
      the kernel, returning the result directly.

      Example (eager mode)::

          @tilelang.jit
          def gemm(A, B, C, block_M: int = 64):
              M, N, K = T.const("M N K")
              A: T.Tensor[[M, K], dtype]  # tensor shape via annotation
              B: T.Tensor[[K, N], dtype]
              C: T.Tensor[[M, N], dtype]
              with T.Kernel(...):
                  ...

          gemm(A, B, C)  # compiles and executes immediately

    The mode is automatically inferred based on whether the function returns a
    PrimFunc (lazy) or uses the builder pattern (eager).

    Attributes
    ----------
    out_idx : list[int] | int | None
        Index(es) of output tensor(s) to return (lazy mode only).
    execution_backend : str | None
        Backend for kernel execution ("auto", "tvm_ffi", etc.).
    target : str | tvm.target.Target | None
        TVM compilation target (e.g., "cuda", "llvm", "auto").
    target_host : str | tvm.target.Target | None
        Host target for cross-compilation.
    verbose : bool | None
        Enable verbose compilation output.
    pass_configs : dict[str, Any] | None
        TVM pass configuration options.
    debug_root_path : str | None
        Directory to save compiled kernel source for debugging.
    compile_flags : list[str] | str | None
        Additional compiler flags.
    func_source : str
        Original Python source code of the decorated function.
    signature : inspect.Signature
        Function signature of the original function.
    mode : Literal["auto", "lazy", "eager"]
        Execution mode. "auto" infers from function behavior.
    func : JITFunc
        The wrapped function object.
    """

    out_idx: list[int] | int | None
    execution_backend: Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "simulator"] | None
    target: TargetLike | None
    target_host: TargetLike | None
    verbose: bool | None
    pass_configs: dict[str, Any] | None
    debug_root_path: str | None
    compile_flags: list[str] | str | None
    func_source: str
    signature: inspect.Signature
    mode: Literal["auto", "lazy", "eager"]
    # place func at the last element for better __repr__
    func: JITFunc[_KP, _T]

    def __post_init__(self):
        if self.debug_root_path is not None and not path.isabs(self.debug_root_path):
            try:
                base_path = path.dirname(path.dirname(path.dirname(__file__)))
                self.debug_root_path = path.join(base_path, self.debug_root_path)
            except NameError:
                self.debug_root_path = path.abspath(self.debug_root_path)
        self._kernel_cache: dict[tuple, Kernel] = {}
        # Enclosing PassContext each _kernel_cache entry was compiled under.
        # Absent means "injected, matches any context" (see _lookup_kernel_cache).
        self._kernel_cache_contexts: dict[tuple, _OuterConfigKey] = {}
        # Kernels compiled for the same argument key under different enclosing
        # PassContexts. Keyed the same way as _kernel_cache so that cache stays
        # a plain argument-keyed map (see _lookup_kernel_cache).
        self._kernel_cache_variants: dict[tuple, dict[_OuterConfigKey, Kernel]] = {}
        self._call_form_cache: _CallFormCache = _CallFormCache()
        self._tuner_cache: dict[tuple, Kernel] = {}

    def get_tir(self, *args: _P.args, **kwargs: _P.kwargs) -> PrimFunc[_KP, _T]:
        """
        Retrieve a TIR (Tensor Intermediate Representation) PrimFunc from the stored callable or object.
        """
        self.initialize_jit_mode(*args, **kwargs)
        if isinstance(self.func, PrimFunc):
            tir = self.func
        elif callable(self.func):
            tir = self.func(*args, **kwargs)
        else:
            raise ValueError(f"Invalid function type: {type(self.func)}")
        assert isinstance(tir, PrimFunc), f"target function must be a PrimFunc but got {type(tir)}"
        return tir

    def _infer_jit_mode(self, *args: _P.args, **kwargs: _P.kwargs) -> Literal["lazy", "eager"]:
        """
        Infer the JIT execution mode based on function behavior.

        Returns "lazy" if the function explicitly returns a PrimFunc,
        or "eager" if it uses the DSL builder pattern.
        """
        if self.mode in ("lazy", "eager"):
            return self.mode
        # auto: infer by checking if function returns PrimFunc directly
        if not isinstance(self.func, JITFunc):
            return "lazy"
        is_lazy_style = self.func._is_lazy_style(*args, **kwargs)
        return "lazy" if is_lazy_style else "eager"

    def initialize_jit_mode(self, *args: _P.args, **kwargs: _P.kwargs) -> Literal["lazy", "eager"]:
        if self.mode == "auto":
            self.mode = self._infer_jit_mode(*args, **kwargs)
        self.func.set_mode(self.mode)
        if self.mode == "eager" and self.out_idx is not None:
            raise ValueError("out_idx is only supported in lazy mode. In eager mode, use T.empty() to declare output tensors instead.")
        return self.mode

    def par_compile(
        self,
        configs: Iterable[dict[str, Any] | tuple[str, Any]],
        num_workers: int = None,
        ignore_error: bool = False,
    ) -> list[JITKernel[_KP, _T]]:
        """
        Parallel compile multiple TileLang PrimFunc with TVM and build JITKernels.
        Parameters
        ----------
        configs : Iterable[Union[dict[str, Any], tuple[Any, ...]]]
            The configurations to elaborate and compile. Each config can be either
            a dictionary mapping keyword arguments to values, or a tuple of positional
            arguments.
        num_workers : int, optional
            Number of parallel workers to use for compilation. Defaults to None,
            which lets the system decide.
        ignore_error : bool, optional
            If True, compilation errors for individual configs will be logged
            as warnings and the corresponding result will be None. If False,
            any compilation error will raise an exception. Defaults to False.
        Returns
        -------
        List[JITKernel]
            A list of compiled JITKernel objects corresponding to the provided configs.
        """

        configs = list(configs)
        funcs = []
        for cfg in tqdm(configs, desc="Elaborating"):
            if isinstance(cfg, tuple):
                funcs.append(self.get_tir(*cfg))
            elif isinstance(cfg, dict):
                funcs.append(self.get_tir(**cfg))
            else:
                raise ValueError(f"Invalid config type: {type(cfg)}, expected tuple or dict.")
        return par_compile(
            funcs,
            out_idx=self.out_idx,
            execution_backend=self.execution_backend,
            target=self.target,
            target_host=self.target_host,
            verbose=self.verbose,
            pass_configs=self.pass_configs,
            compile_flags=self.compile_flags,
            num_workers=num_workers,
            ignore_error=ignore_error,
        )

    def _lookup_kernel_cache(self, key: tuple) -> Kernel | None:
        """Look up a kernel compiled for ``key`` under the enclosing PassContext.

        ``key`` comes from ``parse_args``, so it covers argument shapes/dtypes but
        not the enclosing PassContext. The same call under two different contexts
        is two different kernels: without distinguishing them the first
        compilation wins and every later ``with PassContext(config=...)`` around
        an unchanged call is silently ignored.

        ``_kernel_cache`` therefore stays a plain argument-keyed map holding the
        kernel for the context it was first compiled under, and any additional
        contexts go to ``_kernel_cache_variants``. Keeping the primary map keyed
        on arguments alone means a kernel injected directly into it (tests, or
        anything pre-seeding the cache) is still found, rather than being missed
        because it lacks the context half of a composite key.
        """
        kernel = self._kernel_cache.get(key, None)
        if kernel is None:
            return None

        outer = _outer_pass_config_key()
        recorded = self._kernel_cache_contexts.get(key, _KERNEL_CACHE_ANY_CONTEXT)
        # _KERNEL_CACHE_ANY_CONTEXT: entry was injected rather than compiled
        # here, so there is no context to disagree with.
        if recorded is _KERNEL_CACHE_ANY_CONTEXT or recorded == outer:
            return kernel
        return self._kernel_cache_variants.get(key, {}).get(outer, None)

    def _store_kernel_cache(self, key: tuple, kernel: Kernel) -> None:
        """Record ``kernel`` for ``key`` under the enclosing PassContext."""
        outer = _outer_pass_config_key()
        if key not in self._kernel_cache:
            self._kernel_cache[key] = kernel
            self._kernel_cache_contexts[key] = outer
        else:
            self._kernel_cache_variants.setdefault(key, {})[outer] = kernel

    def compile(self, *args: _P.args, **kwargs: _P.kwargs) -> _Ret:
        prim_func = self.get_tir(*args, **kwargs)
        kernel_result = compile(
            prim_func,
            out_idx=self.out_idx,
            execution_backend=self.execution_backend,
            target=self.target,
            target_host=self.target_host,
            verbose=self.verbose,
            pass_configs=self.pass_configs,
            compile_flags=self.compile_flags,
        )

        if self.debug_root_path:
            if isinstance(self.func, PrimFunc):
                func_name = self.func.attrs["global_symbol"]
            else:
                func_name = getattr(self.func, "__name__", "jit_kernel")

            # cutedsl emits python executor not `c`
            is_cutedsl = (self.execution_backend or self.target) == "cutedsl"
            kernel_suffix = "py" if is_cutedsl else "c"
            kernel_file = f"tilelang_jit_kernel_{func_name}.{kernel_suffix}"

            program_file = f"tilelang_jit_program_{func_name}.py"
            makedirs(self.debug_root_path, exist_ok=True)
            with open(path.join(self.debug_root_path, kernel_file), "w") as f:
                print(kernel_result.get_kernel_source(), file=f)
            with open(path.join(self.debug_root_path, program_file), "w") as f:
                print(prim_func.script(), file=f)

        return kernel_result

    def parse_cache_key(self, *args: _P.args, **kwargs: _P.kwargs):
        tune_params = kwargs.pop("__tune_params", {})
        key_args_tuple = args
        key_kwargs_tuple = tuple(sorted(kwargs.items()))
        tuned_key_kwargs_tuple = tuple(sorted(tune_params.items()))
        key = (key_args_tuple, key_kwargs_tuple, tuned_key_kwargs_tuple)
        return key

    def get_kernel_source(self, *args: _P.args, **kwargs: _P.kwargs) -> str:
        kernel = self.compile(*args, **kwargs)
        return kernel.get_kernel_source()

    def is_lazy_mode(self) -> bool:
        return self.mode == "lazy"

    def _can_use_call_form_cache(self, has_tune_params: bool) -> bool:
        # This cache returns a kernel object directly, so it is only valid for
        # JIT functions that have no runtime tensor arguments to extract.
        return not has_tune_params and isinstance(self.func, JITFunc) and not self.func.tensor_args

    def _gemm_pad_enabled(self):
        pc = self.pass_configs or {}
        return bool(pc.get("tl.enable_gemm_pad", False))

    def _gemm_pad_sizes(self):
        pc = self.pass_configs or {}

        def _int(key):
            v = pc.get(key, 0)
            return int(v) if v else 0

        return _int("tl.gemm_pad_m"), _int("tl.gemm_pad_n"), _int("tl.gemm_pad_k")

    def _try_pad_gemm(self, args, kwargs, pad_m, pad_n, pad_k):
        try:
            bound = self.func._argument_binder.bind(args, kwargs)
            ck = bound.compile_kwargs
        except Exception:
            return None
        found = {}
        for key, val in ck.items():
            kl = key.lower()
            if kl in ("m", "n", "k") and kl not in found:
                found[kl] = (key, val)
        if "m" not in found or "n" not in found or "k" not in found:
            return None
        new_kwargs = dict(ck)
        new_kwargs[found["m"][0]] = found["m"][1] + pad_m
        new_kwargs[found["n"][0]] = found["n"][1] + pad_n
        new_kwargs[found["k"][0]] = found["k"][1] + pad_k
        kernel = self.compile(**new_kwargs)
        return _PaddedKernel(kernel, pad_m, pad_n, pad_k, found["m"][1], found["n"][1])

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _Ret:
        # Separate out the tuning parameters from the user's kwargs
        # Whether to return the compile arguments (out_idx, target, target_host, etc.) for autotuner cache
        return_compile_arguments = kwargs.pop("__return_compile_arguments", False)
        if return_compile_arguments:
            logger.warning("`__return_compile_arguments` is deprecated and will be removed in future versions.")
            compile_args = {
                "out_idx": self.out_idx,
                "execution_backend": self.execution_backend,
                "target": self.target,
                "target_host": self.target_host,
                "verbose": self.verbose,
                "pass_configs": self.pass_configs,
                "compile_flags": self.compile_flags,
            }
            return compile_args

        has_tune_params = "__tune_params" in kwargs
        kwargs.update(kwargs.pop("__tune_params", {}))

        # infer mode early, before parse_args needs it
        if self.mode == "auto":
            self.mode = self._infer_jit_mode(*args, **kwargs)
            self.func.set_mode(self.mode)

        # Data-layer padding: gated by tl.enable_gemm_pad, sized by tl.gemm_pad_m/n/k
        if self._gemm_pad_enabled():
            pad_m, pad_n, pad_k = self._gemm_pad_sizes()
            if (pad_m or pad_n or pad_k) and self.is_lazy_mode():
                padded = self._try_pad_gemm(args, kwargs, pad_m, pad_n, pad_k)
                if padded is not None:
                    return padded

        call_form_key = None
        if self.is_lazy_mode() and self._can_use_call_form_cache(has_tune_params):
            kernel, call_form_key = self._call_form_cache.lookup(args, kwargs)
            if kernel is not _CALL_FORM_CACHE_MISS:
                return kernel

        key, kernel_args = self.func.parse_args(*args, **kwargs)
        kernel = self._lookup_kernel_cache(key)
        if kernel is None:
            kernel = self.compile(*args, **kwargs)
            self._store_kernel_cache(key, kernel)

        if call_form_key is not None and self.is_lazy_mode() and not kernel_args:
            self._call_form_cache.store(call_form_key, kernel)

        # eager mode: execute kernel immediately and return result
        # lazy mode: return kernel object for manual invocation
        if self.mode == "eager":
            return kernel(*kernel_args.values())
        else:
            return kernel


ExecutionBackend = Literal["auto", "dlpack", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "simulator"]


@overload
def jit(func: Callable[_KP, _T]) -> JITImpl[_KP, _KP, _T, _T]: ...


@overload
def jit(
    *,
    out_idx: Any = None,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    execution_backend: ExecutionBackend | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    debug_root_path: str | None = None,
    compile_flags: list[str] | str | None = None,
) -> Callable[[Callable[_KP, _T]], JITImpl[_KP, _KP, _T, _T]]: ...


def jit(
    func: Callable[_P, _T] | PrimFunc | None = None,
    *,  # Indicates subsequent arguments are keyword-only
    out_idx: list[int] | int | None = None,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    execution_backend: ExecutionBackend | None = None,
    verbose: bool | None = None,
    pass_configs: dict[str, Any] | None = None,
    debug_root_path: str | None = None,
    compile_flags: list[str] | str | None = None,
) -> Callable[[Callable[_P, _T]], JITImpl[_KP, _KP, _T, _T]]:
    """
    JIT compiler decorator for TileLang functions.

    Supports two execution modes (automatically inferred):
    - **lazy**: Function returns PrimFunc explicitly. Returns compiled kernel object.
    - **eager**: Function uses DSL builder pattern. Executes kernel immediately.

    Parameters
    ----------
    out_idx : list[int] | int | None
        Output tensor index(es). Only supported in lazy mode.
    target : str | tvm.target.Target | None
        Compilation target (e.g., "cuda", "llvm", "auto").
    target_host : str | tvm.target.Target | None
        Host target for cross-compilation.
    execution_backend : ExecutionBackend | None
        Backend for kernel execution.
    verbose : bool | None
        Enable verbose compilation output.
    pass_configs : dict[str, Any] | None
        TVM pass configuration options.
    debug_root_path : str | None
        Directory to save compiled kernel source for debugging.
    compile_flags : list[str] | str | None
        Additional compiler flags.
    """

    compile_args = dict(
        out_idx=out_idx,
        execution_backend=execution_backend,
        target=target,
        target_host=target_host,
        verbose=verbose,
        pass_configs=pass_configs,
        debug_root_path=debug_root_path,
        compile_flags=compile_flags,
    )

    def decorator(func: Callable[_P, _T]):
        mode = "auto"
        pf: JITFunc[_P, _T] = prim_func(func, eager_jit=True)
        func_source = inspect.getsource(pf.orig_func)
        signature = inspect.signature(pf.orig_func)

        return JITImpl(
            func=pf,
            **compile_args,
            func_source=func_source,
            signature=signature,
            mode=mode,
        )

    return decorator(func) if func is not None else decorator
