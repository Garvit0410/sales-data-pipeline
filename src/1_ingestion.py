# Step 1: read the three source files and save them as raw tables
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

import common

ORDER_COLUMNS = ["row_id", "order_id", "order_date", "ship_date", "ship_mode",
                 "customer_id", "product_id", "quantity", "price", "discount", "profit"]
CUSTOMER_COLUMNS = ["customer_id", "customer_name", "email", "phone", "address",
                    "segment", "country", "city", "state", "postal_code", "region"]
PRODUCT_COLUMNS = ["product_id", "category", "sub_category", "product_name",
                   "state", "price_per_product"]


def check_file(path):
    # fail early with a clear message if the file is not there
    if not Path(path).exists():
        raise FileNotFoundError(f"input file not found: {path}")


def check_columns(df_columns, expected, file_name):
    # fail early if the file does not have the columns we need
    missing = [c for c in expected if c not in df_columns]
    if missing:
        raise ValueError(f"{file_name} is missing columns: {missing}")


def to_snake_case(name):
    # "Sub-Category" -> "sub_category"
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def rename_columns(df_input):
    # give every column a snake case name
    for col in df_input.columns:
        df_input = df_input.withColumnRenamed(col, to_snake_case(col))
    return df_input


def to_number(col_name, type_name):
    # a value that is not a number becomes null instead of stopping the load
    return F.expr(f"try_cast(`{col_name}` as {type_name})")


def read_customers(spark, path):
    # the excel file is small, so pandas is enough to read it
    check_file(path)
    df_pandas = pd.read_excel(path, dtype=str)
    df_pandas.columns = [to_snake_case(c) for c in df_pandas.columns]
    check_columns(df_pandas.columns, CUSTOMER_COLUMNS, "customer file")

    # empty cells become real nulls
    rows = [tuple(None if pd.isna(v) else v for v in row)
            for row in df_pandas.itertuples(index=False, name=None)]
    schema = ", ".join(f"{c} string" for c in df_pandas.columns)
    return spark.createDataFrame(rows, schema)


def read_products(spark, path):
    # quotes inside a product name are escaped with another quote
    check_file(path)
    df_products = spark.read.option("header", True).option("multiLine", True).option("escape", '"').csv(path)
    df_products = rename_columns(df_products)
    check_columns(df_products.columns, PRODUCT_COLUMNS, "product file")

    # the price is read as text, so turn it into a number
    return df_products.withColumn("price_per_product", to_number("price_per_product", "double"))


def read_orders(spark, path):
    check_file(path)
    df_orders = rename_columns(spark.read.option("multiLine", True).json(path))
    check_columns(df_orders.columns, ORDER_COLUMNS, "order file")

    # fixed column order, and proper number types
    return (df_orders.select(ORDER_COLUMNS)
              .withColumn("row_id", to_number("row_id", "long"))
              .withColumn("quantity", to_number("quantity", "long"))
              .withColumn("price", to_number("price", "double"))
              .withColumn("discount", to_number("discount", "double"))
              .withColumn("profit", to_number("profit", "double")))


def run(spark):
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {common.DATABASE}")
    common.save_table(read_customers(spark, common.CUSTOMERS_FILE), common.RAW_CUSTOMERS)
    common.save_table(read_products(spark, common.PRODUCTS_FILE), common.RAW_PRODUCTS)
    common.save_table(read_orders(spark, common.ORDERS_FILE), common.RAW_ORDERS)


if __name__ == "__main__":
    run(common.get_spark())
