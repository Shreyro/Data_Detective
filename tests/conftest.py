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
