# Step 2: clean customers and products
from pyspark.sql import functions as F

import common


def clean_string(col):
    # turn any run of spaces/tabs/newlines into one space, then trim the ends
    text = F.trim(F.regexp_replace(col, r"\s+", " "))
    # a value that is empty after tidying becomes null
    return F.when(text == "", F.lit(None)).otherwise(text)


def clean_name(col):
    # drop an apostrophe or hyphen unless it sits between two letters (keeps O'Brien, Smith-Jones)
    name = F.regexp_replace(col, r"(?<!\p{L})['\-]|['\-](?!\p{L})", "")
    # drop anything that is not a letter, space, apostrophe or hyphen (digits and symbols)
    name = F.regexp_replace(name, r"[^\p{L}\s'\-]", "")
    return clean_string(name)


def enrich_customers(df_raw):
    df_customers = (df_raw
                    # rows without a customer id are dropped
                    .withColumn("customer_id", F.trim("customer_id"))
                    .filter(F.col("customer_id").isNotNull() & (F.col("customer_id") != ""))
                    .select(
                        "customer_id",
                        # cleaned name, with the original kept next to it
                        clean_name(F.col("customer_name")).alias("customer_name"),
                        F.col("customer_name").alias("customer_name_raw"),
                        F.lower(clean_string(F.col("email"))).alias("email"),
                        clean_string(F.col("segment")).alias("segment"),
                        clean_string(F.col("country")).alias("country"),
                        clean_string(F.col("city")).alias("city"),
                        clean_string(F.col("state")).alias("state"),
                        # excel drops leading zeros, so pad to 5 digits
                        F.lpad(F.trim(F.col("postal_code")), 5, "0").alias("postal_code"),
                        clean_string(F.col("region")).alias("region"),
                    ))
    # one row per customer
    return df_customers.dropDuplicates(["customer_id"])


def enrich_products(df_raw):
    # a product id repeats once per state, so keep one row per id
    df_products = (df_raw
                   .withColumn("product_id", F.trim("product_id"))
                   .filter(F.col("product_id").isNotNull() & (F.col("product_id") != "")))
    # category and name are the same across states, the price is averaged
    return df_products.groupBy("product_id").agg(
        F.min(clean_string(F.col("category"))).alias("category"),
        F.min(clean_string(F.col("sub_category"))).alias("sub_category"),
        F.min(clean_string(F.col("product_name"))).alias("product_name"),
        F.round(F.avg("price_per_product"), 2).alias("avg_price_per_product"),
        F.countDistinct("state").alias("state_count"),
    )


def run(spark):
    df_raw_customers = spark.table(common.table_name(common.RAW_CUSTOMERS))
    df_raw_products = spark.table(common.table_name(common.RAW_PRODUCTS))
    common.save_table(enrich_customers(df_raw_customers), common.ENRICHED_CUSTOMERS)
    common.save_table(enrich_products(df_raw_products), common.ENRICHED_PRODUCTS)


if __name__ == "__main__":
    run(common.get_spark())
