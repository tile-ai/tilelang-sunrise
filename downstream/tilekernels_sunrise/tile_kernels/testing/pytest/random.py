"""Deterministic random seeding pytest plugin for tile_kernels."""

import hashlib
import random

import pytest
import tile_kernels


def pytest_addoption(parser):
    parser.addoption('--seed', type=int, default=0)


@pytest.fixture(autouse=True)
def seed(request):
    base = request.config.getoption('--seed')
    node_hash = int(hashlib.sha256(request.node.nodeid.encode()).hexdigest(), 16) % (2**31)
    seed = base + node_hash
    random.seed(seed)
    tile_kernels.rand.set_seed(seed)
    return seed
