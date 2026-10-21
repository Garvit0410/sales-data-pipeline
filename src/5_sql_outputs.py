# Step 5: the four profit reports, written in SQL

import common


def profit_by_year(spark, table):
    # total profit for each year
    return spark.sql(f"""
        SELECT order_year, ROUND(SUM(total_profit), 2) AS total_profit
        FROM {table}
        GROUP BY order_year
        ORDER BY order_year
    """)


def profit_by_year_category(spark, table):
    # total profit for each year and product category
    return spark.sql(f"""
        SELECT order_year, category, ROUND(SUM(total_profit), 2) AS total_profit
        FROM {table}
        GROUP BY order_year, category
        ORDER BY order_year, category
    """)


def profit_by_customer(spark, table):
    # total profit for each customer, highest first
    return spark.sql(f"""
        SELECT customer_id, customer_name, ROUND(SUM(total_profit), 2) AS total_profit
        FROM {table}
        GROUP BY customer_id, customer_name
        ORDER BY total_profit DESC, customer_id
    """)


def profit_by_customer_year(spark, table):
    # total profit for each customer and year
    return spark.sql(f"""
        SELECT customer_id, customer_name, order_year, ROUND(SUM(total_profit), 2) AS total_profit
        FROM {table}
        GROUP BY customer_id, customer_name, order_year
        ORDER BY customer_id, order_year
    """)


def run(spark):
    table = common.table_name(common.PROFIT_AGGREGATE)
    reports = {
        "profit_by_year": profit_by_year(spark, table),
        "profit_by_year_category": profit_by_year_category(spark, table),
        "profit_by_customer": profit_by_customer(spark, table),
        "profit_by_customer_year": profit_by_customer_year(spark, table),
    }
    # show each report and save it as a csv
    for name, df_report in reports.items():
        print(name)
        df_report.show(20, truncate=False)
        common.save_csv(df_report, name)
    return reports


if __name__ == "__main__":
    run(common.get_spark())
