# Sales Data Pipeline

PySpark pipeline that turns raw customer, product and order files into clean tables and profit reports. Runs locally and on Databricks Free Edition.

## Project structure

​```
sales-data-pipeline/
├── data/                                  raw input files
│   ├── Customer.xlsx
│   ├── Products.csv
│   └── Orders.json
├── src/                                   pipeline code (run in order 1 to 5)
│   ├── common.py                          paths, table names and helper functions
│   ├── 1_ingestion.py / .ipynb            create raw tables
│   ├── 2_enrich_customers_products.py     clean customers, one row per product
│   ├── 3_enrich_orders.py                 add customer and product details to orders
│   ├── 4_profit_aggregate.py              profit by year, category, sub category, customer
│   └── 5_sql_outputs.py / .ipynb          four SQL reports
├── tests/                                 pytest unit tests
│   ├── conftest.py                        Spark session and sample data
│   ├── test_transformations.py
│   ├── test_aggregations.py
│   ├── test_pipeline_e2e.py
│   └── run_tests.ipynb                    run the tests from Databricks
└── results/                               CSV output of every table and report
​```

## How to run

Run the scripts or notebooks in `src/` in numbered order, then run `pytest tests` to check the results.
