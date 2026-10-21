import os
import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import common

# Make sure PySpark workers use the same interpreter as pytest
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    if common.ON_DATABRICKS:
        # use the session Databricks already gives us
        yield SparkSession.builder.getOrCreate()
        return

    warehouse = tmp_path_factory.mktemp("warehouse")
    session = (
        SparkSession.builder.master("local[2]")
        .appName("sales-pipeline-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.warehouse.dir", str(warehouse))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.ansi.enabled", "true")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return common.DATA_DIR


@pytest.fixture()
def df_raw_customers(spark):
    rows = [
        ("C1", "Tam&^*ara Willing___)ham", "A@X.com", "1", "addr", "Consumer", "United States", "NYC", "New York", "2101", "East"),
        ("C2", "  Jim   Smith ", None, None, None, "Corporate", "United States", "LA", "California", "90001", "West"),
        ("C3", None, None, None, None, "Consumer", "United States", "SF", "California", "94105", "West"),
        ("C1", "Duplicate Of C1", None, None, None, "Consumer", "United States", "NYC", "New York", "2101", "East"),
    ]
    cols = ["customer_id", "customer_name", "email", "phone", "address", "segment",
            "country", "city", "state", "postal_code", "region"]
    return spark.createDataFrame(rows, cols)


@pytest.fixture()
def df_raw_products(spark):
    rows = [
        ("P1", "Furniture", "Chairs", "Chair, Black", "New York", 80.0),
        ("P1", "Furniture", "Chairs", "Chair Black", "Texas", 60.0),
        ("P2", "Technology", "Phones", "Phone", "Ohio", 100.0),
    ]
    cols = ["product_id", "category", "sub_category", "product_name", "state", "price_per_product"]
    return spark.createDataFrame(rows, cols)


@pytest.fixture()
def df_raw_orders(spark):
    rows = [
        (1, "O-1", "21/8/2016", "25/8/2016", "Standard", "C1", "P1", 2, 100.0, 0.0, 10.123456789),
        (2, "O-2", "1/1/2017", "2/1/2017", "First", "C2", "P2", 1, 50.0, 0.1, -3.456),
        (3, "O-3", "5/12/2017", "9/12/2017", "First", "C3", "PX", 1, 20.0, 0.0, 5.0),
        (4, "O-4", "15/3/2016", "16/3/2016", "Same Day", "C1", "P2", 3, 30.0, 0.0, 0.125),
    ]
    cols = ["row_id", "order_id", "order_date", "ship_date", "ship_mode", "customer_id",
            "product_id", "quantity", "price", "discount", "profit"]
    return spark.createDataFrame(rows, cols)
