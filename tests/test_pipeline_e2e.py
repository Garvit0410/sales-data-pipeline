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

ORDER_ROW = {"Row ID": 1, "Order ID": "O1", "Order Date": "1/1/2016", "Ship Date": "2/1/2016",
             "Ship Mode": "Standard", "Customer ID": "C1", "Product ID": "P1",
             "Quantity": 1, "Price": 10.0, "Discount": 0.0, "Profit": 5.5}
PRODUCT_HEADER = "Product ID,Category,Sub-Category,Product Name,State,Price per product\n"


def write_orders(scratch_dir, rows):
    path = scratch_dir / "Orders.json"
    path.write_text(json.dumps(rows))
    return str(path)


def write_products(scratch_dir, text):
    path = scratch_dir / "Products.csv"
    path.write_text(text)
    return str(path)


def write_customers(scratch_dir, columns, postal_code="x"):
    row = [postal_code if c == "postal_code" else "C1" if c == "customer_id" else "x" for c in columns]
    path = scratch_dir / "Customer.xlsx"
    pd.DataFrame([row], columns=columns).to_excel(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def df_source_customers(data_dir):
    return pd.read_excel(data_dir / "Customer.xlsx", dtype=str)


@pytest.fixture(scope="module")
def df_source_products(data_dir):
    return pd.read_csv(data_dir / "Products.csv")


@pytest.fixture(scope="module")
def df_source_orders(data_dir):
    with open(data_dir / "Orders.json") as f:
        return pd.DataFrame(json.load(f))


def test_read_customers(spark, data_dir, df_source_customers):
    df_customers = ingestion.read_customers(spark, str(data_dir / "Customer.xlsx"))
    assert df_customers.count() == len(df_source_customers)
    assert df_customers.columns == ["customer_id", "customer_name", "email", "phone", "address",
                                    "segment", "country", "city", "state", "postal_code", "region"]
    expected_null_names = df_source_customers["Customer Name"].isna().sum()
    assert df_customers.filter(F.col("customer_name").isNull()).count() == expected_null_names
    # everything is read as text so leading zeros are not lost
    assert {t for _, t in df_customers.dtypes} == {"string"}


def test_read_products(spark, data_dir, df_source_products):
    df_products = ingestion.read_products(spark, str(data_dir / "Products.csv"))
    assert df_products.count() == len(df_source_products)
    assert df_products.columns == ingestion.PRODUCT_COLUMNS
    types = dict(df_products.dtypes)
    assert types.pop("price_per_product") == "double"
    assert set(types.values()) == {"string"}

    chair = df_products.filter("product_id = 'FUR-CH-10002961' AND state = 'New York'").first()
    assert chair["product_name"] == "Leather Task Chair, Black"
    assert chair["price_per_product"] == pytest.approx(81.882)

    binder = df_products.filter("product_id = 'OFF-BI-10004233'").first()
    assert '8-1/2" x 11"' in binder["product_name"]


def test_read_orders(spark, data_dir, df_source_orders):
    df_orders = ingestion.read_orders(spark, str(data_dir / "Orders.json"))
    assert df_orders.count() == len(df_source_orders)
    assert df_orders.columns == ingestion.ORDER_COLUMNS
    # full schema check
    assert dict(df_orders.dtypes) == {
        "row_id": "bigint", "order_id": "string", "order_date": "string", "ship_date": "string",
        "ship_mode": "string", "customer_id": "string", "product_id": "string",
        "quantity": "bigint", "price": "double", "discount": "double", "profit": "double"}


def test_to_snake_case():
    assert ingestion.to_snake_case("Sub-Category") == "sub_category"
    assert ingestion.to_snake_case("Price per product") == "price_per_product"
    assert ingestion.to_snake_case(" Row ID ") == "row_id"
    assert ingestion.to_snake_case("  Sub-Category ") == "sub_category"


# ---------- negative cases for ingestion ----------

@pytest.mark.parametrize("reader, name", [
    ("read_customers", "nope.xlsx"), ("read_products", "nope.csv"), ("read_orders", "nope.json")])
def test_missing_file_raises_file_not_found(spark, scratch_dir, reader, name):
    with pytest.raises(FileNotFoundError, match=name):
        getattr(ingestion, reader)(spark, str(scratch_dir / name))


def test_order_file_without_profit_raises_value_error(spark, scratch_dir):
    row = {k: v for k, v in ORDER_ROW.items() if k != "Profit"}
    with pytest.raises(ValueError, match="profit"):
        ingestion.read_orders(spark, write_orders(scratch_dir, [row]))


def test_empty_order_file_raises_value_error(spark, scratch_dir):
    with pytest.raises(ValueError, match="missing columns"):
        ingestion.read_orders(spark, write_orders(scratch_dir, []))


def test_product_file_without_price_raises_value_error(spark, scratch_dir):
    path = write_products(scratch_dir, "Product ID,Category,Sub-Category,Product Name,State\nP1,A,B,C,D\n")
    with pytest.raises(ValueError, match="price_per_product"):
        ingestion.read_products(spark, path)


def test_customer_file_without_id_raises_value_error(spark, scratch_dir):
    columns = [c for c in ingestion.CUSTOMER_COLUMNS if c != "customer_id"]
    with pytest.raises(ValueError, match="customer_id"):
        ingestion.read_customers(spark, write_customers(scratch_dir, columns))


@pytest.mark.parametrize("column, bad_value", [
    ("Profit", "abc"), ("Price", "$10"), ("Discount", "ten percent"),
    ("Quantity", "two"), ("Row ID", "x1")])
def test_bad_number_becomes_null_and_row_is_kept(spark, scratch_dir, column, bad_value):
    bad_row = dict(ORDER_ROW, **{"Row ID": 2, column: bad_value})
    df_orders = ingestion.read_orders(spark, write_orders(scratch_dir, [ORDER_ROW, bad_row]))
    assert df_orders.count() == 2
    assert df_orders.filter(f"{ingestion.to_snake_case(column)} IS NULL").count() == 1


def test_bad_price_in_products_becomes_null(spark, scratch_dir):
    path = write_products(scratch_dir, PRODUCT_HEADER + "P1,A,B,C,D,80.5\nP2,A,B,C,D,N/A\n")
    prices = {r["product_id"]: r["price_per_product"] for r in ingestion.read_products(spark, path).collect()}
    assert prices == {"P1": 80.5, "P2": None}


def test_null_values_in_orders_are_kept(spark, scratch_dir):
    row = dict(ORDER_ROW, Profit=None)
    assert ingestion.read_orders(spark, write_orders(scratch_dir, [row])).first()["profit"] is None


def test_extra_columns_in_orders_are_ignored(spark, scratch_dir):
    row = dict(ORDER_ROW, Notes="extra")
    assert ingestion.read_orders(spark, write_orders(scratch_dir, [row])).columns == ingestion.ORDER_COLUMNS


def test_customer_leading_zero_is_kept(spark, scratch_dir):
    path = write_customers(scratch_dir, ingestion.CUSTOMER_COLUMNS, postal_code="02101")
    assert ingestion.read_customers(spark, path).first()["postal_code"] == "02101"


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


def test_all_tables_exist(spark, run_all):
    for name in [common.RAW_CUSTOMERS, common.RAW_PRODUCTS, common.RAW_ORDERS,
                 common.ENRICHED_CUSTOMERS, common.ENRICHED_PRODUCTS,
                 common.ENRICHED_ORDERS, common.PROFIT_AGGREGATE]:
        assert spark.catalog.tableExists(f"{DATABASE}.{name}"), name


def test_raw_tables_keep_all_rows(spark, run_all, df_source_customers, df_source_products, df_source_orders):
    assert table(spark, common.RAW_CUSTOMERS).count() == len(df_source_customers)
    assert table(spark, common.RAW_PRODUCTS).count() == len(df_source_products)
    assert table(spark, common.RAW_ORDERS).count() == len(df_source_orders)


def test_enriched_keys_are_unique(spark, run_all, df_source_customers, df_source_products):
    df_customers = table(spark, common.ENRICHED_CUSTOMERS)
    df_products = table(spark, common.ENRICHED_PRODUCTS)
    # one row per id, and no id is lost
    assert df_customers.count() == df_customers.select("customer_id").distinct().count()
    assert df_customers.count() == df_source_customers["Customer ID"].str.strip().nunique()
    assert df_products.count() == df_products.select("product_id").distinct().count()
    assert df_products.count() == df_source_products["Product ID"].str.strip().nunique()


def test_customer_names_are_clean(spark, run_all):
    df_customers = table(spark, common.ENRICHED_CUSTOMERS)
    df_not_clean = df_customers.filter(
        F.col("customer_name").isNotNull()
        & ~F.col("customer_name").rlike(r"^[\p{L}'\-]+( [\p{L}'\-]+)*$"))
    assert df_not_clean.count() == 0


def test_enriched_orders_keep_every_order_once(spark, run_all, df_source_orders):
    df_orders = table(spark, common.ENRICHED_ORDERS)
    expected = df_source_orders["Row ID"].nunique()
    assert df_orders.count() == expected
    assert df_orders.select("row_id").distinct().count() == expected


def test_enriched_profit_matches_source(spark, run_all, df_source_orders):
    df_got = table(spark, common.ENRICHED_ORDERS).select("row_id", "profit").toPandas()
    got = df_got.set_index("row_id")["profit"].sort_index()
    expected = df_source_orders.set_index("Row ID")["Profit"].round(2).sort_index()
    assert (got - expected).abs().max() < 1e-9


def test_orders_with_missing_details(spark, run_all, df_source_orders, df_source_products):
    df_orders = table(spark, common.ENRICHED_ORDERS)
    # orders whose product is not in the product file have no category
    no_product = ~df_source_orders["Product ID"].isin(df_source_products["Product ID"])
    assert df_orders.filter("category IS NULL").count() == no_product.sum()

    # orders whose customer is missing, or has no usable name, have no customer name
    names = {r["customer_id"]: r["customer_name"]
             for r in table(spark, common.ENRICHED_CUSTOMERS).collect()}
    no_name = df_source_orders["Customer ID"].map(lambda c: names.get(c) is None)
    assert df_orders.filter("customer_name IS NULL").count() == no_name.sum()


def test_aggregate_total_matches_source(spark, run_all, df_source_orders):
    total = table(spark, common.PROFIT_AGGREGATE).agg(F.sum("total_profit")).first()[0]
    assert total == pytest.approx(df_source_orders["Profit"].round(2).sum(), abs=0.01)


def test_aggregate_years_and_categories(spark, run_all, df_source_orders, df_source_products):
    df_aggregate = table(spark, common.PROFIT_AGGREGATE)
    years = sorted(r[0] for r in df_aggregate.select("order_year").distinct().collect())
    categories = sorted(r[0] for r in df_aggregate.select("category").distinct().collect())
    order_dates = pd.to_datetime(df_source_orders["Order Date"], format="%d/%m/%Y")
    assert years == sorted(order_dates.dt.year.unique())
    # every product category, plus Unknown for orders without a product
    expected = set(df_source_products["Category"].dropna()) | {"Unknown"}
    assert categories == sorted(expected)


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


def test_running_again_does_not_add_rows(spark, run_all, df_source_orders):
    run_all()
    assert table(spark, common.RAW_ORDERS).count() == len(df_source_orders)
    assert table(spark, common.ENRICHED_ORDERS).count() == df_source_orders["Row ID"].nunique()


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
