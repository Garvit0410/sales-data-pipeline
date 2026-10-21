# Step 4: profit by year, category, sub category and customer
from pyspark.sql import functions as F

import common


def build_profit_aggregate(df_orders, unknown_label="Unknown"):
    return (df_orders
            # orders with no matching product are grouped as "Unknown" so the totals still add up
            .withColumn("category", F.coalesce("category", F.lit(unknown_label)))
            .withColumn("sub_category", F.coalesce("sub_category", F.lit(unknown_label)))
            # total profit for each year, category, sub category and customer
            .groupBy("order_year", "category", "sub_category", "customer_id", "customer_name")
            .agg(F.round(F.sum("profit"), 2).alias("total_profit")))


def run(spark):
    df_orders = spark.table(common.table_name(common.ENRICHED_ORDERS))
    common.save_table(build_profit_aggregate(df_orders), common.PROFIT_AGGREGATE)


if __name__ == "__main__":
    run(common.get_spark())
