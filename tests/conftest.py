import pytest

from aiops.config import small_config
from aiops.detect import RollingZScore, SeasonalResidual
from aiops.pipeline import build
from aiops.simulate import simulate


@pytest.fixture(scope="session")
def cfg():
    return small_config(seed=7)


@pytest.fixture(scope="session")
def dataset(cfg):
    return simulate(cfg)


@pytest.fixture(scope="session")
def built(cfg):
    # Two fast detectors are enough to exercise every later stage.
    return build(cfg, detectors=[RollingZScore(), SeasonalResidual()])


@pytest.fixture(scope="session")
def workspace(built):
    return built.workspace
