import importlib

import pytest
from pyspark.sql import functions as F

enrich = importlib.import_module("2_enrich_customers_products")
enrich_orders = importlib.import_module("3_enrich_orders")


def apply_to_value(spark, value, func):
    df_value = spark.createDataFrame([(value,)], "v string")
    return df_value.select(func(F.col("v")).alias("r")).first()["r"]


@pytest.mark.parametrize("dirty, expected", [
    ("Tam&^*ara Willing___)ham", "Tamara Willingham"),
    ("Gary567 Hansen", "Gary Hansen"),
    ("   _Mike Vitt 12313orini", "Mike Vitt orini"),
    ("Ad.       ..am Hart", "Ad am Hart"),
    ("  Jim   Smith ", "Jim Smith"),
    ("Jason Fortune-", "Jason Fortune"),
    ("&^*(5678Shirley Daniels", "Shirley Daniels"),
    ("Neil Knudson%^&*(", "Neil Knudson"),
    ("Peter Bühler", "Peter Bühler"),
    ("Patrick O'Brien", "Patrick O'Brien"),
    ("Mary Smith-Jones", "Mary Smith-Jones"),
    ("Kristi;'[]na Nunn", "Kristina Nunn"),
    ("Anna Smith", "Anna Smith"),
])
def test_clean_name(spark, dirty, expected):
    assert apply_to_value(spark, dirty, enrich.clean_name) == expected


@pytest.mark.parametrize("value", [None, "", "   ", "123 456", "@#$%"])
def test_clean_name_gives_null_when_nothing_is_left(spark, value):
    assert apply_to_value(spark, value, enrich.clean_name) is None


def test_clean_name_twice_gives_same_result(spark):
    once = apply_to_value(spark, "Tam&^*ara Willing___)ham", enrich.clean_name)
    assert apply_to_value(spark, once, enrich.clean_name) == once


def test_clean_string(spark):
    assert apply_to_value(spark, "  a   b ", enrich.clean_string) == "a b"
    assert apply_to_value(spark, "   ", enrich.clean_string) is None
    assert apply_to_value(spark, None, enrich.clean_string) is None


def test_customers_have_one_row_per_id(df_raw_customers):
    df_out = enrich.enrich_customers(df_raw_customers)
    assert df_out.count() == 3
    assert df_out.select("customer_id").distinct().count() == 3


def test_customer_ids_with_extra_spaces_count_as_one(spark, df_raw_customers):
    # " C2 " is the same customer as "C2"
    df_spaced = df_raw_customers.withColumn(
        "customer_id", F.when(F.col("customer_id") == "C2", " C2 ").otherwise(F.col("customer_id")))
    df_both = df_spaced.unionByName(df_raw_customers.filter("customer_id = 'C2'"))
    assert enrich.enrich_customers(df_both).filter("customer_id = 'C2'").count() == 1


def test_customers_are_cleaned(df_raw_customers):
    rows = {r["customer_id"]: r for r in enrich.enrich_customers(df_raw_customers).collect()}
    assert rows["C2"]["customer_name"] == "Jim Smith"
    assert rows["C3"]["customer_name"] is None
    assert rows["C1"]["customer_name_raw"] is not None
    assert rows["C1"]["postal_code"] == "02101"
    assert rows["C1"]["email"] == "a@x.com"


def test_customers_without_id_are_dropped(spark):
    other_columns = ["email", "segment", "country", "city", "state", "postal_code", "region"]
    df_raw = spark.createDataFrame(
        [(None, "X"), ("  ", "Y"), ("C9", "Z")], "customer_id string, customer_name string")
    for col in other_columns:
        df_raw = df_raw.withColumn(col, F.lit(None).cast("string"))
    assert [r["customer_id"] for r in enrich.enrich_customers(df_raw).collect()] == ["C9"]


def test_products_have_one_row_per_id(df_raw_products):
    df_out = enrich.enrich_products(df_raw_products)
    assert df_out.count() == 2
    assert df_out.select("product_id").distinct().count() == 2


def test_product_ids_with_extra_spaces_count_as_one(df_raw_products):
    df_spaced = df_raw_products.withColumn(
        "product_id", F.when(F.col("state") == "Texas", " P1 ").otherwise(F.col("product_id")))
    assert enrich.enrich_products(df_spaced).filter("product_id = 'P1'").count() == 1


def test_product_values(df_raw_products):
    rows = {r["product_id"]: r for r in enrich.enrich_products(df_raw_products).collect()}
    p1 = rows["P1"]
    assert (p1["category"], p1["sub_category"]) == ("Furniture", "Chairs")
    assert p1["avg_price_per_product"] == 70.0
    assert p1["state_count"] == 2
    assert p1["product_name"] == "Chair Black"


@pytest.fixture()
def df_enriched(df_raw_orders, df_raw_customers, df_raw_products):
    return enrich_orders.build_enriched_orders(
        df_raw_orders,
        enrich.enrich_customers(df_raw_customers),
        enrich.enrich_products(df_raw_products),
    )


def test_order_count_is_unchanged(df_enriched, df_raw_orders):
    # product P1 appears twice in the products, it must not repeat the order
    assert df_enriched.count() == df_raw_orders.count() == 4


def test_duplicate_orders_are_removed(df_raw_orders, df_raw_customers, df_raw_products):
    df_out = enrich_orders.build_enriched_orders(
        df_raw_orders.unionByName(df_raw_orders),
        enrich.enrich_customers(df_raw_customers),
        enrich.enrich_products(df_raw_products),
    )
    assert df_out.count() == 4


def test_same_row_id_with_different_values_keeps_one_row(df_raw_orders, df_raw_customers, df_raw_products):
    df_changed = df_raw_orders.filter("row_id = 1").withColumn("profit", F.lit(999.0))
    df_out = enrich_orders.build_enriched_orders(
        df_raw_orders.unionByName(df_changed),
        enrich.enrich_customers(df_raw_customers),
        enrich.enrich_products(df_raw_products),
    )
    assert df_out.filter("row_id = 1").count() == 1
    assert df_out.count() == 4


def test_duplicate_customers_in_lookup_do_not_repeat_orders(spark, df_raw_orders, df_raw_products):
    # the lookup is passed in with a repeated customer id on purpose
    df_lookup = spark.createDataFrame(
        [("C1", "Ann", "US"), ("C1", "Ann Again", "US"), ("C2", "Bob", "US"), ("C3", "Cy", "US")],
        "customer_id string, customer_name string, country string")
    df_out = enrich_orders.build_enriched_orders(
        df_raw_orders, df_lookup, enrich.enrich_products(df_raw_products))
    assert df_out.count() == 4


def test_customer_ids_with_spaces_in_orders_still_match(df_raw_orders, df_raw_customers, df_raw_products):
    df_spaced = df_raw_orders.withColumn("customer_id", F.concat(F.lit(" "), "customer_id", F.lit(" ")))
    df_out = enrich_orders.build_enriched_orders(
        df_spaced, enrich.enrich_customers(df_raw_customers), enrich.enrich_products(df_raw_products))
    assert df_out.filter("row_id = 1").first()["customer_name"] == "Tamara Willingham"


def test_orders_without_row_id_are_dropped(spark, df_raw_orders, df_raw_customers, df_raw_products):
    df_empty_row = spark.createDataFrame([(None,) * 11], df_raw_orders.schema)
    df_out = enrich_orders.build_enriched_orders(
        df_raw_orders.unionByName(df_empty_row),
        enrich.enrich_customers(df_raw_customers),
        enrich.enrich_products(df_raw_products),
    )
    assert df_out.count() == 4


def test_enriched_order_columns(df_enriched):
    assert df_enriched.columns == [
        "row_id", "order_id", "order_date", "order_year", "ship_date", "ship_mode",
        "customer_id", "customer_name", "customer_country",
        "product_id", "category", "sub_category",
        "quantity", "price", "discount", "profit",
    ]


def test_profit_is_rounded_to_2_decimals(df_enriched):
    profits = {r["row_id"]: r["profit"] for r in df_enriched.collect()}
    assert profits == {1: 10.12, 2: -3.46, 3: 5.0, 4: 0.13}


def test_customer_and_product_details(df_enriched):
    rows = {r["row_id"]: r for r in df_enriched.collect()}
    assert rows[1]["customer_name"] == "Tamara Willingham"
    assert rows[1]["customer_country"] == "United States"
    assert (rows[1]["category"], rows[1]["sub_category"]) == ("Furniture", "Chairs")
    assert (rows[2]["category"], rows[2]["sub_category"]) == ("Technology", "Phones")


def test_unknown_product_is_kept(df_enriched):
    row = df_enriched.filter("row_id = 3").first()
    assert row["category"] is None and row["sub_category"] is None
    assert row["profit"] == 5.0


def test_dates_are_read_day_first(df_enriched):
    rows = {r["row_id"]: r for r in df_enriched.collect()}
    assert {k: v["order_year"] for k, v in rows.items()} == {1: 2016, 2: 2017, 3: 2017, 4: 2016}
    assert str(rows[1]["order_date"]) == "2016-08-21"
    assert str(rows[4]["order_date"]) == "2016-03-15"


def test_bad_date_becomes_null(df_raw_orders, df_raw_customers, df_raw_products):
    df_bad = df_raw_orders.withColumn(
        "order_date", F.when(F.col("row_id") == 1, "not-a-date").otherwise(F.col("order_date")))
    df_out = enrich_orders.build_enriched_orders(
        df_bad, enrich.enrich_customers(df_raw_customers), enrich.enrich_products(df_raw_products))
    row = df_out.filter("row_id = 1").first()
    assert row["order_date"] is None and row["order_year"] is None
    assert df_out.count() == 4


def test_unknown_customer_is_kept(spark, df_raw_orders, df_raw_customers, df_raw_products):
    df_extra = spark.createDataFrame(
        [(5, "O-5", "1/1/2015", "2/1/2015", "First", "NOPE", "P1", 1, 1.0, 0.0, 1.0)],
        df_raw_orders.schema)
    df_out = enrich_orders.build_enriched_orders(
        df_raw_orders.unionByName(df_extra),
        enrich.enrich_customers(df_raw_customers),
        enrich.enrich_products(df_raw_products),
    )
    row = df_out.filter("row_id = 5").first()
    assert row["customer_name"] is None and row["customer_country"] is None
    assert df_out.count() == 5
