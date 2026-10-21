# Step 3: orders with customer and product details, profit rounded to 2 decimals
from pyspark.sql import functions as F

import common

DATE_FORMAT = "d/M/yyyy"


def build_enriched_orders(df_raw_orders, df_customers, df_products):
    # keep valid orders only, one row per row_id
    df_orders = df_raw_orders.filter(F.col("row_id").isNotNull()).dropDuplicates(["row_id"])

    # one row per key so the joins can never repeat an order
    df_customers = (df_customers
                    .select("customer_id", "customer_name", F.col("country").alias("customer_country"))
                    .dropDuplicates(["customer_id"]))
    df_products = (df_products
                   .select("product_id", "category", "sub_category")
                   .dropDuplicates(["product_id"]))

    # try_to_timestamp gives null for a bad date instead of stopping with an error
    order_date = F.try_to_timestamp(F.col("order_date"), F.lit(DATE_FORMAT)).cast("date")
    ship_date = F.try_to_timestamp(F.col("ship_date"), F.lit(DATE_FORMAT)).cast("date")

    # left joins keep orders even when the customer or product is not found
    return (df_orders
            .withColumn("customer_id", F.trim("customer_id"))
            .withColumn("product_id", F.trim("product_id"))
            .join(df_customers, "customer_id", "left")
            .join(df_products, "product_id", "left")
            .select(
                "row_id", "order_id",
                order_date.alias("order_date"),
                # the year comes from the order date
                F.year(order_date).alias("order_year"),
                ship_date.alias("ship_date"),
                "ship_mode",
                "customer_id", "customer_name", "customer_country",
                "product_id", "category", "sub_category",
                "quantity", "price", "discount",
                # profit rounded to 2 decimals
                F.round("profit", 2).alias("profit"),
            ))


def run(spark):
    df_orders = build_enriched_orders(
        spark.table(common.table_name(common.RAW_ORDERS)),
        spark.table(common.table_name(common.ENRICHED_CUSTOMERS)),
        spark.table(common.table_name(common.ENRICHED_PRODUCTS)),
    )
    common.save_table(df_orders, common.ENRICHED_ORDERS)


if __name__ == "__main__":
    run(common.get_spark())
