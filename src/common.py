# Settings shared by all the steps
import os
from pathlib import Path

from pyspark.sql import SparkSession

ON_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ or os.path.exists("/databricks")

ROOT = Path(__file__).resolve().parents[1]

# on Databricks the data and results live in a volume, change this to your own volume
BASE_DIR = Path("/Volumes/workspace/default/sales_project") if ON_DATABRICKS else ROOT
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

CUSTOMERS_FILE = str(DATA_DIR / "Customer.xlsx")
PRODUCTS_FILE = str(DATA_DIR / "Products.csv")
ORDERS_FILE = str(DATA_DIR / "Orders.json")

DATABASE = "sales"

# delta on Databricks, plain parquet when running on your own machine
TABLE_FORMAT = "delta" if ON_DATABRICKS else "parquet"

RAW_CUSTOMERS = "raw_customers"
RAW_PRODUCTS = "raw_products"
RAW_ORDERS = "raw_orders"
ENRICHED_CUSTOMERS = "enriched_customers"
ENRICHED_PRODUCTS = "enriched_products"
ENRICHED_ORDERS = "enriched_orders"
PROFIT_AGGREGATE = "profit_aggregate"


def get_spark():
    if ON_DATABRICKS:
        return SparkSession.builder.getOrCreate()

    # hive support keeps the tables between runs, so each step can be run on its own
    # the tables are kept in a hidden .spark folder inside the project
    spark_dir = ROOT / ".spark"
    return (SparkSession.builder.appName("sales_pipeline")
            .config("spark.sql.warehouse.dir", str(spark_dir / "warehouse"))
            .config("spark.hadoop.javax.jdo.option.ConnectionURL",
                    f"jdbc:derby:;databaseName={spark_dir / 'metastore_db'};create=true")
            .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={spark_dir}")
            .enableHiveSupport()
            .getOrCreate())


def table_name(name):
    return DATABASE + "." + name


def save_csv(df_result, name):
    # one plain csv file per result, saved in the results folder
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_result.toPandas().to_csv(RESULTS_DIR / (name + ".csv"), index=False)


def save_table(df_result, name):
    df_result.write.format(TABLE_FORMAT).mode("overwrite").saveAsTable(table_name(name))
    save_csv(df_result, name)
