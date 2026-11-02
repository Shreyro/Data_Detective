import os

import pytest

from data_detective import datagen
from data_detective.agent import DataDetective
from data_detective.db import Warehouse
from data_detective.offline import OfflineBrain


@pytest.fixture(scope="session")
def wh(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "test.duckdb"
    datagen.build(path)
    return Warehouse(path)


@pytest.fixture(scope="session")
def agent(wh):
    return DataDetective(wh, OfflineBrain())


@pytest.fixture(autouse=True, scope="session")
def _no_api_key():
    """The whole suite must pass without an API key (CI has none)."""
    old = os.environ.pop("ANTHROPIC_API_KEY", None)
    yield
    if old is not None:
        os.environ["ANTHROPIC_API_KEY"] = old
