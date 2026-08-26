"""Base classes for workload definitions shared between tests and benchmarks.

WorkloadBase defines the contract: gen_inputs() for input generation.
FixtureMeta / FixtureBase provide reusable pytest parametrize decorators.

Correctness-only logic (ref_program, check, tolerances) stays in tests/.
"""

from __future__ import annotations

import contextlib
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

import torch

_F = TypeVar("_F", bound=Callable[..., Any])

# Random factories that accept ``device=`` and draw from a generator. Each is
# redirected to the CPU inside ``seeded_rng`` so seeding actually takes effect.
_SEEDED_FACTORIES = (
    "randn",
    "rand",
    "randint",
    "randperm",
    "normal",
    "bernoulli",
)

#: Seed used for workload input generation. Matches the value the pytest
#: ``setup`` fixture passes to ``torch.manual_seed``.
WORKLOAD_SEED = 1235


@contextlib.contextmanager
def seeded_rng(seed: int = WORKLOAD_SEED):
    """Make ``torch.randn(..., device="ptpu")`` and friends reproducible.

    ``torch_ptpu`` ships no RNG API -- no ``manual_seed``, ``manual_seed_all``,
    or ``default_generators`` -- so ``torch.manual_seed`` cannot reach the PTPU
    generator and torch warns that the seed "does not take effect". Workloads
    that build inputs directly on the device therefore get fresh data on every
    run, and data that depends on how much randomness ran before them. A failure
    is then only reproducible by re-running the exact same set of test cases,
    and a nondeterministic kernel is indistinguishable from a flaky test.

    Inside this context each factory in :data:`_SEEDED_FACTORIES` draws on the
    CPU -- whose RNG does honour seeding -- from one generator shared across the
    whole block, then moves the result to the requested device. Sharing a single
    generator (rather than reseeding per call) keeps successive tensors distinct,
    so q/k/v do not collapse to identical data. Calls that pass an explicit
    ``generator=`` are left alone.

    Factories are patched on the ``torch`` module, so this is not thread-safe;
    workload input generation is single-threaded.
    """
    gen = torch.Generator().manual_seed(seed)
    originals = {}

    def wrap(orig):
        @functools.wraps(orig)
        def wrapper(*args, **kwargs):
            device = kwargs.get("device")
            if kwargs.get("generator") is not None or device is None:
                return orig(*args, **kwargs)
            if not str(device).startswith(("ptpu", "cuda")):
                return orig(*args, **kwargs)
            kwargs = dict(kwargs)
            kwargs.pop("device")
            kwargs["generator"] = gen
            return orig(*args, **kwargs).to(device)

        return wrapper

    try:
        for name in _SEEDED_FACTORIES:
            orig = getattr(torch, name, None)
            if orig is None:
                continue
            originals[name] = orig
            setattr(torch, name, wrap(orig))
        yield gen
    finally:
        for name, orig in originals.items():
            setattr(torch, name, orig)


class WorkloadBase(ABC):
    """Abstract base for workload definitions (input generation + parameters).

    Subclass must implement gen_inputs().
    Used by both tests (via TestBase) and benchmarks (via BenchmarkBase).

    Correctness-only methods (ref_program, check, tolerances) belong in
    tests/ — not here.

    Every subclass's ``gen_inputs`` is wrapped in :func:`seeded_rng` so inputs
    are reproducible without each of the ~65 implementations having to thread a
    generator through by hand.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        fn = cls.__dict__.get("gen_inputs")
        if fn is None or getattr(fn, "_seeded", False):
            return

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            with seeded_rng():
                return fn(self, *args, **kwargs)

        wrapper._seeded = True
        cls.gen_inputs = wrapper

    @abstractmethod
    def gen_inputs(self) -> tuple[Any, ...]:
        raise NotImplementedError


class RandnTest(WorkloadBase):
    """Workload base for ops whose inputs are generated via ``torch.randn``."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="ptpu")
        return (x,)


class FixtureMeta(type):
    """Metaclass that makes Fixture subclasses usable as @decorators.

    Usage:
        class MyFixture(FixtureBase):
            @classmethod
            def get_params(cls):
                import pytest
                return [("a, b", [
                    pytest.param(1, 2, marks=pytest.mark.smoke),
                ])]

        @MyFixture
        def test_something(a, b): ...

    PARAMS may also be set as a plain class variable (list) for backwards
    compatibility when pytest is already importable at module scope.
    """

    def __call__(cls, fn: _F) -> _F:
        import pytest  # lazy import: pytest is only needed when applying parametrize decorators

        params = cls.get_params() if hasattr(cls, "get_params") else cls.PARAMS
        for names, values in reversed(params):
            fn = pytest.mark.parametrize(names, values)(fn)
        return fn


class FixtureBase(metaclass=FixtureMeta):
    """Base class for reusable parametrize decorators.

    Subclass and set PARAMS (plain list) or override get_params() (classmethod
    that lazily imports pytest) to provide a list of (names_str, values_list)
    tuples.
    - Single entry with multiple param names -> explicit combinations
    - Multiple entries each with one param name -> cross-product
    """
    PARAMS = []
