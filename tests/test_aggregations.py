import importlib

import pytest
from pyspark.sql import functions as F

enrich = importlib.import_module("2_enrich_customers_products")
enrich_orders = importlib.import_module("3_enrich_orders")
aggregate_step = importlib.import_module("4_profit_aggregate")
sql_step = importlib.import_module("5_sql_outputs")

VIEW = "t_profit_aggregate"


@pytest.fixture()
def df_enriched_orders(spark):
    columns = ("row_id long, order_year int, customer_id string, customer_name string, "
               "category string, sub_category string, profit double")
    rows = [
        (1, 2016, "C1", "Ann", "Furniture", "Chairs", 10.10),
        (2, 2016, "C1", "Ann", "Furniture", "Chairs", 5.20),
        (3, 2016, "C1", "Ann", "Technology", "Phones", -2.00),
        (4, 2017, "C1", "Ann", "Furniture", "Chairs", 1.00),
        (5, 2017, "C2", "Bob", "Furniture", "Tables", 7.50),
        (6, 2017, "C2", "Bob", None, None, 3.00),
        (7, 2016, "C3", None, "Technology", "Phones", 0.30),
    ]
    return spark.createDataFrame(rows, columns)


@pytest.fixture()
def df_aggregate(df_enriched_orders):
    return aggregate_step.build_profit_aggregate(df_enriched_orders)


def to_dict(df_result, keys):
    return {tuple(r[k] for k in keys): r["total_profit"] for r in df_result.collect()}


def test_aggregate_columns_and_grain(df_aggregate):
    assert df_aggregate.columns == ["order_year", "category", "sub_category",
                                    "customer_id", "customer_name", "total_profit"]
    keys = ["order_year", "category", "sub_category", "customer_id"]
    assert df_aggregate.count() == df_aggregate.select(*keys).distinct().count()


def test_aggregate_sums_each_group(df_aggregate):
    got = to_dict(df_aggregate, ["order_year", "category", "sub_category", "customer_id"])
    assert got[(2016, "Furniture", "Chairs", "C1")] == 15.3
    assert got[(2016, "Technology", "Phones", "C1")] == -2.0
    assert got[(2017, "Furniture", "Tables", "C2")] == 7.5


def test_aggregate_total_matches_orders(df_aggregate, df_enriched_orders):
    total = df_aggregate.agg(F.round(F.sum("total_profit"), 2)).first()[0]
    assert total == df_enriched_orders.agg(F.round(F.sum("profit"), 2)).first()[0] == 25.1


def test_unknown_product_is_labelled(df_aggregate):
    row = df_aggregate.filter("customer_id = 'C2' AND order_year = 2017 AND category = 'Unknown'").first()
    assert row["sub_category"] == "Unknown" and row["total_profit"] == 3.0


def test_custom_unknown_label(df_enriched_orders):
    df_out = aggregate_step.build_profit_aggregate(df_enriched_orders, unknown_label="N/A")
    assert df_out.filter("category = 'N/A'").count() == 1


def test_customer_without_name_is_kept(df_aggregate):
    row = df_aggregate.filter("customer_id = 'C3'").first()
    assert row["customer_name"] is None and row["total_profit"] == 0.3


def test_empty_input_gives_empty_aggregate(df_enriched_orders):
    assert aggregate_step.build_profit_aggregate(df_enriched_orders.limit(0)).count() == 0


@pytest.fixture()
def reports(spark, df_aggregate):
    df_aggregate.createOrReplaceTempView(VIEW)
    return {
        "year": sql_step.profit_by_year(spark, VIEW),
        "year_category": sql_step.profit_by_year_category(spark, VIEW),
        "customer": sql_step.profit_by_customer(spark, VIEW),
        "customer_year": sql_step.profit_by_customer_year(spark, VIEW),
    }


def test_profit_by_year(reports):
    assert to_dict(reports["year"], ["order_year"]) == {(2016,): 13.6, (2017,): 11.5}


def test_profit_by_year_and_category(reports):
    assert to_dict(reports["year_category"], ["order_year", "category"]) == {
        (2016, "Furniture"): 15.3, (2016, "Technology"): -1.7,
        (2017, "Furniture"): 8.5, (2017, "Unknown"): 3.0,
    }


def test_profit_by_customer(reports):
    assert to_dict(reports["customer"], ["customer_id"]) == {
        ("C1",): 14.3, ("C2",): 10.5, ("C3",): 0.3}


def test_profit_by_customer_and_year(reports):
    assert to_dict(reports["customer_year"], ["customer_id", "order_year"]) == {
        ("C1", 2016): 13.3, ("C1", 2017): 1.0, ("C2", 2017): 10.5, ("C3", 2016): 0.3}


def test_all_reports_have_the_same_total(reports):
    totals = {round(df_report.agg(F.sum("total_profit")).first()[0], 2) for df_report in reports.values()}
    assert totals == {25.1}


def test_profit_by_customer_is_sorted_high_to_low(reports):
    values = [r["total_profit"] for r in reports["customer"].collect()]
    assert values == sorted(values, reverse=True)


def test_duplicate_orders_do_not_double_the_profit(df_raw_orders, df_raw_customers, df_raw_products):
    def total_profit(df_raw):
        df_orders = enrich_orders.build_enriched_orders(
            df_raw, enrich.enrich_customers(df_raw_customers), enrich.enrich_products(df_raw_products))
        df_out = aggregate_step.build_profit_aggregate(df_orders)
        return df_out.agg(F.round(F.sum("total_profit"), 2)).first()[0]

    assert total_profit(df_raw_orders.unionByName(df_raw_orders)) == total_profit(df_raw_orders)


def test_reports_have_no_repeated_rows(reports):
    keys = {"year": ["order_year"], "year_category": ["order_year", "category"],
            "customer": ["customer_id"], "customer_year": ["customer_id", "order_year"]}
    for name, columns in keys.items():
        df_report = reports[name]
        assert df_report.count() == df_report.select(*columns).distinct().count(), name
