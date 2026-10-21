import importlib
import json

import pandas as pd
import pytest
from pyspark.sql import functions as F

import common

ingestion = importlib.import_module("1_ingestion")
enrich = importlib.import_module("2_enrich_customers_products")
enrich_orders = importlib.import_module("3_enrich_orders")
aggregate_step = importlib.import_module("4_profit_aggregate")
sql_step = importlib.import_module("5_sql_outputs")

DATABASE = "e2e_sales"


def test_read_customers(spark, data_dir):
    df_customers = ingestion.read_customers(spark, str(data_dir / "Customer.xlsx"))
    assert df_customers.count() == 793
    assert df_customers.columns == ["customer_id", "customer_name", "email", "phone", "address",
                                    "segment", "country", "city", "state", "postal_code", "region"]
    assert df_customers.filter(F.col("customer_name").isNull()).count() == 8


def test_read_products(spark, data_dir):
    df_products = ingestion.read_products(spark, str(data_dir / "Products.csv"))
    assert df_products.count() == 1851
    assert df_products.columns == ["product_id", "category", "sub_category", "product_name",
                                   "state", "price_per_product"]

    chair = df_products.filter("product_id = 'FUR-CH-10002961' AND state = 'New York'").first()
    assert chair["product_name"] == "Leather Task Chair, Black"
    assert chair["price_per_product"] == pytest.approx(81.882)

    binder = df_products.filter("product_id = 'OFF-BI-10004233'").first()
    assert '8-1/2" x 11"' in binder["product_name"]


def test_read_orders(spark, data_dir):
    df_orders = ingestion.read_orders(spark, str(data_dir / "Orders.json"))
    assert df_orders.count() == 9994
    assert df_orders.columns == ingestion.ORDER_COLUMNS
    types = dict(df_orders.dtypes)
    assert types["row_id"] == "bigint"
    assert types["profit"] == "double"
    assert types["order_date"] == "string"


def test_to_snake_case():
    assert ingestion.to_snake_case("Sub-Category") == "sub_category"
    assert ingestion.to_snake_case("Price per product") == "price_per_product"
    assert ingestion.to_snake_case(" Row ID ") == "row_id"


@pytest.fixture(scope="module")
def run_all(spark, data_dir, tmp_path_factory):
    # point the settings at the real files and a throwaway database
    patch = pytest.MonkeyPatch()
    patch.setattr(common, "CUSTOMERS_FILE", str(data_dir / "Customer.xlsx"))
    patch.setattr(common, "PRODUCTS_FILE", str(data_dir / "Products.csv"))
    patch.setattr(common, "ORDERS_FILE", str(data_dir / "Orders.json"))
    patch.setattr(common, "DATABASE", DATABASE)
    patch.setattr(common, "RESULTS_DIR", tmp_path_factory.mktemp("results"))

    def run_steps():
        ingestion.run(spark)
        enrich.run(spark)
        enrich_orders.run(spark)
        aggregate_step.run(spark)
        return sql_step.run(spark)

    spark.sql(f"DROP DATABASE IF EXISTS {DATABASE} CASCADE")
    run_steps.reports = run_steps()
    yield run_steps
    spark.sql(f"DROP DATABASE IF EXISTS {DATABASE} CASCADE")
    patch.undo()


def table(spark, name):
    return spark.table(f"{DATABASE}.{name}")


@pytest.fixture(scope="module")
def df_source_orders(data_dir):
    with open(data_dir / "Orders.json") as f:
        return pd.DataFrame(json.load(f))


def test_all_tables_exist(spark, run_all):
    for name in [common.RAW_CUSTOMERS, common.RAW_PRODUCTS, common.RAW_ORDERS,
                 common.ENRICHED_CUSTOMERS, common.ENRICHED_PRODUCTS,
                 common.ENRICHED_ORDERS, common.PROFIT_AGGREGATE]:
        assert spark.catalog.tableExists(f"{DATABASE}.{name}"), name


def test_raw_tables_keep_all_rows(spark, run_all):
    assert table(spark, common.RAW_CUSTOMERS).count() == 793
    assert table(spark, common.RAW_PRODUCTS).count() == 1851
    assert table(spark, common.RAW_ORDERS).count() == 9994


def test_enriched_keys_are_unique(spark, run_all):
    df_customers = table(spark, common.ENRICHED_CUSTOMERS)
    df_products = table(spark, common.ENRICHED_PRODUCTS)
    assert df_customers.count() == df_customers.select("customer_id").distinct().count() == 793
    assert df_products.count() == df_products.select("product_id").distinct().count() == 1818


def test_customer_names_are_clean(spark, run_all):
    df_customers = table(spark, common.ENRICHED_CUSTOMERS)
    df_not_clean = df_customers.filter(
        F.col("customer_name").isNotNull()
        & ~F.col("customer_name").rlike(r"^[\p{L}'\-]+( [\p{L}'\-]+)*$"))
    assert df_not_clean.count() == 0


def test_enriched_orders_keep_every_order_once(spark, run_all):
    df_orders = table(spark, common.ENRICHED_ORDERS)
    assert df_orders.count() == 9994
    assert df_orders.select("row_id").distinct().count() == 9994


def test_enriched_profit_matches_source(spark, run_all, df_source_orders):
    df_got = table(spark, common.ENRICHED_ORDERS).select("row_id", "profit").toPandas()
    got = df_got.set_index("row_id")["profit"].sort_index()
    expected = df_source_orders.set_index("Row ID")["Profit"].round(2).sort_index()
    assert (got - expected).abs().max() < 1e-9


def test_orders_with_missing_details(spark, run_all):
    df_orders = table(spark, common.ENRICHED_ORDERS)
    assert df_orders.filter("category IS NULL").count() == 204
    assert df_orders.filter("customer_name IS NULL").count() == 137


def test_aggregate_total_matches_source(spark, run_all, df_source_orders):
    total = table(spark, common.PROFIT_AGGREGATE).agg(F.sum("total_profit")).first()[0]
    assert total == pytest.approx(df_source_orders["Profit"].round(2).sum(), abs=0.01)


def test_aggregate_years_and_categories(spark, run_all):
    df_aggregate = table(spark, common.PROFIT_AGGREGATE)
    years = sorted(r[0] for r in df_aggregate.select("order_year").distinct().collect())
    categories = sorted(r[0] for r in df_aggregate.select("category").distinct().collect())
    assert years == [2014, 2015, 2016, 2017]
    assert categories == ["Furniture", "Office Supplies", "Technology", "Unknown"]


def test_reports_match_pandas(run_all, df_source_orders, data_dir):
    reports = run_all.reports
    df_orders = df_source_orders.copy()
    df_orders["year"] = pd.to_datetime(df_orders["Order Date"], format="%d/%m/%Y").dt.year
    df_orders["Profit"] = df_orders["Profit"].round(2)
    df_products = pd.read_csv(data_dir / "Products.csv").drop_duplicates("Product ID")
    df_orders = df_orders.merge(df_products[["Product ID", "Category"]], on="Product ID", how="left")
    df_orders = df_orders.fillna({"Category": "Unknown"})

    by_year = {r["order_year"]: r["total_profit"]
               for r in reports["profit_by_year"].collect()}
    assert by_year == pytest.approx(df_orders.groupby("year")["Profit"].sum().round(2).to_dict(), abs=0.011)

    by_year_category = {(r["order_year"], r["category"]): r["total_profit"]
                        for r in reports["profit_by_year_category"].collect()}
    expected = df_orders.groupby(["year", "Category"])["Profit"].sum().round(2).to_dict()
    assert by_year_category == pytest.approx(expected, abs=0.011)

    by_customer = {r["customer_id"]: r["total_profit"]
                   for r in reports["profit_by_customer"].collect()}
    expected = df_orders.groupby("Customer ID")["Profit"].sum().round(2).to_dict()
    assert by_customer == pytest.approx(expected, abs=0.011)

    by_customer_year = {(r["customer_id"], r["order_year"]): r["total_profit"]
                        for r in reports["profit_by_customer_year"].collect()}
    expected = df_orders.groupby(["Customer ID", "year"])["Profit"].sum().round(2).to_dict()
    assert by_customer_year == pytest.approx(expected, abs=0.011)


def test_running_again_does_not_add_rows(spark, run_all):
    run_all()
    assert table(spark, common.RAW_ORDERS).count() == 9994
    assert table(spark, common.ENRICHED_ORDERS).count() == 9994


def test_every_table_and_report_is_saved_as_csv(spark, run_all):
    tables = [common.RAW_CUSTOMERS, common.RAW_PRODUCTS, common.RAW_ORDERS,
              common.ENRICHED_CUSTOMERS, common.ENRICHED_PRODUCTS,
              common.ENRICHED_ORDERS, common.PROFIT_AGGREGATE]
    for name in tables:
        df_saved = pd.read_csv(common.RESULTS_DIR / (name + ".csv"))
        assert len(df_saved) == table(spark, name).count(), name

    aggregate = f"{DATABASE}.{common.PROFIT_AGGREGATE}"
    reports = {
        "profit_by_year": sql_step.profit_by_year(spark, aggregate),
        "profit_by_year_category": sql_step.profit_by_year_category(spark, aggregate),
        "profit_by_customer": sql_step.profit_by_customer(spark, aggregate),
        "profit_by_customer_year": sql_step.profit_by_customer_year(spark, aggregate),
    }
    for name, df_report in reports.items():
        df_saved = pd.read_csv(common.RESULTS_DIR / (name + ".csv"))
        assert len(df_saved) == df_report.count(), name


def test_saved_csv_keeps_text_values(run_all):
    df_customers = pd.read_csv(common.RESULTS_DIR / (common.ENRICHED_CUSTOMERS + ".csv"), dtype=str)
    assert df_customers["postal_code"].str.len().eq(5).all()
    assert not df_customers["customer_name"].dropna().str.contains(r"[0-9&^*_]").any()
