# Step 1: read the three source files and save them as raw tables
import pandas as pd
from pyspark.sql import functions as F

import common

ORDER_COLUMNS = ["row_id", "order_id", "order_date", "ship_date", "ship_mode",
                 "customer_id", "product_id", "quantity", "price", "discount", "profit"]


def to_snake_case(name):
    # "Sub-Category" -> "sub_category"
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def rename_columns(df_input):
    # give every column a snake case name
    for col in df_input.columns:
        df_input = df_input.withColumnRenamed(col, to_snake_case(col))
    return df_input


def read_customers(spark, path):
    # the excel file is small, so pandas is enough to read it
    df_pandas = pd.read_excel(path, dtype=str)
    df_pandas.columns = [to_snake_case(c) for c in df_pandas.columns]

    # empty cells become real nulls
    rows = [tuple(None if pd.isna(v) else v for v in row)
            for row in df_pandas.itertuples(index=False, name=None)]
    schema = ", ".join(f"{c} string" for c in df_pandas.columns)
    return spark.createDataFrame(rows, schema)


def read_products(spark, path):
    # quotes inside a product name are escaped with another quote
    df_products = spark.read.option("header", True).option("multiLine", True).option("escape", '"').csv(path)
    df_products = rename_columns(df_products)

    # the price is read as text, so turn it into a number
    return df_products.withColumn("price_per_product", F.col("price_per_product").cast("double"))


def read_orders(spark, path):
    df_orders = rename_columns(spark.read.option("multiLine", True).json(path))

    # fixed column order, and proper number types
    return (df_orders.select(ORDER_COLUMNS)
              .withColumn("row_id", F.col("row_id").cast("long"))
              .withColumn("quantity", F.col("quantity").cast("long"))
              .withColumn("price", F.col("price").cast("double"))
              .withColumn("discount", F.col("discount").cast("double"))
              .withColumn("profit", F.col("profit").cast("double")))


def run(spark):
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {common.DATABASE}")
    common.save_table(read_customers(spark, common.CUSTOMERS_FILE), common.RAW_CUSTOMERS)
    common.save_table(read_products(spark, common.PRODUCTS_FILE), common.RAW_PRODUCTS)
    common.save_table(read_orders(spark, common.ORDERS_FILE), common.RAW_ORDERS)


if __name__ == "__main__":
    run(common.get_spark())
