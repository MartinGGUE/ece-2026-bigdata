"""Lab Spark DataFrame (bronze to silver) - exercises 1 to 4."""

import datetime
import os
import re
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

BUCKET = os.environ["KUBERNETES_NAMESPACE"]

spark = (
    SparkSession.builder
    .appName("lab-silver-exercises")
    .master("local[*]")
    .config("spark.hadoop.fs.s3a.endpoint", f"https://{os.environ['AWS_S3_ENDPOINT']}")
    .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"])
    .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"])
    .config("spark.hadoop.fs.s3a.session.token", os.environ.get("AWS_SESSION_TOKEN", ""))
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.lab", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.lab.type", "hadoop")
    .config("spark.sql.catalog.lab.warehouse", f"s3a://{BUCKET}/warehouse")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")


def timed(label, func):
    """Run func, print its duration, return its result."""
    start = time.perf_counter()
    result = func()
    print(f"{label:<45} {time.perf_counter() - start:.3f}s")
    return result


def bench(df, runs=3):
    """Best time over several runs to fully compute a DataFrame (the noop sink computes every column)."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        df.write.format("noop").mode("overwrite").save()
        times.append(time.perf_counter() - start)
    return min(times)


# ---------------------------------------------------------------------------
# Setup: rebuild the silver layer exactly as in the lab
# ---------------------------------------------------------------------------
def read_bronze():
    options = {"header": True, "inferSchema": True, "samplingRatio": 0.01, "multiLine": True}
    users = spark.read.options(**options).csv(f"s3a://{BUCKET}/bronze/users.csv")
    orders = spark.read.options(**options).csv(f"s3a://{BUCKET}/bronze/orders.csv")
    return users, orders


def build_users_typed(users_bronze):
    return (
        users_bronze
        .select(
            F.col("uuid").cast("string").alias("user_id"),
            F.trim("username").alias("username"),
            F.trim("name").alias("name"),
            F.upper(F.trim("sex")).alias("sex"),
            F.lower(F.trim("mail")).alias("email"),
            F.col("birthdate").cast("date").alias("birthdate"),
            F.split(F.col("address"), "\n").alias("address_lines"),
        )
        .withColumn("street", F.element_at("address_lines", 1))
        .withColumn("address_line_2", F.element_at("address_lines", 2))
        .drop("address_lines")
    )


def build_stg_users(users_typed):
    stg = users_typed.select(
        "user_id", "username", "name", "sex", "email", "birthdate", "street",
        F.regexp_extract("address_line_2", "^(.+), [A-Z]{2} [0-9]{5}$", 1).alias("city"),
        F.regexp_extract("address_line_2", "([A-Z]{2}) ([0-9]{5})$", 1).alias("state"),
        F.regexp_extract("address_line_2", "([A-Z]{2}) ([0-9]{5})$", 2).alias("zip_code"),
    )
    return stg.withColumn("city", F.when(F.col("city") == "", None).otherwise(F.col("city")))


def build_stg_orders(orders_bronze):
    typed = orders_bronze.select(
        F.col("uuid").cast("string").alias("order_id"),
        F.col("user_uuid").cast("string").alias("user_id"),
        F.col("date").cast("timestamp").alias("ordered_at"),
        F.col("quantity").cast("int").alias("quantity"),
        F.lower(F.trim("product")).alias("product"),
    )
    window = Window.partitionBy("order_id").orderBy(F.col("ordered_at").desc())
    return typed.withColumn("rn", F.row_number().over(window)).filter("rn = 1").drop("rn")


users_bronze, orders_bronze = read_bronze()
users_typed = build_users_typed(users_bronze)
stg_users = build_stg_users(users_typed)
stg_orders = build_stg_orders(orders_bronze)

# ---------------------------------------------------------------------------
# Exercise 1: birthdate_is_valid, with the DataFrame API and with Spark SQL
# ---------------------------------------------------------------------------
print("\n=== Exercise 1: birthdate_is_valid ===")

# DataFrame API: first order of each user, joined to the users
first_orders = stg_orders.groupBy("user_id").agg(F.min("ordered_at").alias("first_order_at"))
users_valid_df = (
    stg_users.join(first_orders, "user_id", "left")
    .withColumn(
        "birthdate_is_valid",
        # A user without order or without birthdate is considered valid
        F.coalesce(F.col("birthdate").cast("timestamp") <= F.col("first_order_at"), F.lit(True)),
    )
    .drop("first_order_at")
)

# Spark SQL: the same logic
stg_users.createOrReplaceTempView("stg_users")
stg_orders.createOrReplaceTempView("stg_orders")
users_valid_sql = spark.sql("""
    WITH first_orders AS (
        SELECT user_id, min(ordered_at) AS first_order_at
        FROM stg_orders
        GROUP BY user_id
    )
    SELECT u.*,
           coalesce(CAST(u.birthdate AS TIMESTAMP) <= f.first_order_at, true) AS birthdate_is_valid
    FROM stg_users u
    LEFT JOIN first_orders f ON u.user_id = f.user_id
""")

agree = (
    users_valid_df.exceptAll(users_valid_sql).count() == 0
    and users_valid_sql.exceptAll(users_valid_df).count() == 0
)
print("DataFrame and SQL versions agree:", agree)
users_valid_df.groupBy("birthdate_is_valid").count().show()
print("Examples of invalid birthdates:")
users_valid_df.filter(~F.col("birthdate_is_valid")).select("username", "birthdate").show(5)
orders_before_birth = (
    stg_orders.join(stg_users, "user_id")
    .filter(F.col("ordered_at") < F.col("birthdate").cast("timestamp"))
    .count()
)
print("Orders placed before the user's birth (dbt test assert_orders_after_user_birth):", orders_before_birth)

# ---------------------------------------------------------------------------
# Exercise 2: zip_code with regexp_extract, a Python UDF and a pandas_udf
# ---------------------------------------------------------------------------
print("\n=== Exercise 2: regexp_extract vs Python UDF vs pandas_udf ===")
ZIP_PATTERN = r"([A-Z]{2}) ([0-9]{5})$"
ZIP_RE = re.compile(ZIP_PATTERN)


@F.udf(returnType=StringType())
def zip_python_udf(line):
    if line is None:
        return None
    match = ZIP_RE.search(line)
    return match.group(2) if match else ""


try:
    import pandas as pd
    import pyarrow  # noqa: F401  (required by pandas_udf)

    @F.pandas_udf(StringType())
    def zip_pandas_udf(lines: pd.Series) -> pd.Series:
        return lines.str.extract(ZIP_PATTERN)[1].fillna("")
except ImportError as error:
    zip_pandas_udf = None
    print("pandas_udf skipped, missing library:", error)

versions = {
    "regexp_extract (built-in)": F.regexp_extract("address_line_2", ZIP_PATTERN, 2),
    "Python UDF": zip_python_udf("address_line_2"),
}
if zip_pandas_udf is not None:
    versions["pandas_udf"] = zip_pandas_udf("address_line_2")

users_typed.cache().count()  # read S3 once, so that only the extraction is timed
reference = sorted(r[0] for r in users_typed.select(versions["regexp_extract (built-in)"]).collect())
for name, expression in versions.items():
    df = users_typed.select(expression.alias("zip_code"))
    same = sorted(r[0] for r in df.collect()) == reference
    print(f"{name:<30} best of 3: {bench(df):.3f}s   same result as regexp_extract: {same}")

# ---------------------------------------------------------------------------
# Exercise 3: cache the bronze DataFrames
# ---------------------------------------------------------------------------
print("\n=== Exercise 3: cache ===")
for label, use_cache in [("without cache", False), ("with cache", True)]:
    users_b, orders_b = read_bronze()
    if use_cache:
        users_b = users_b.cache()
        orders_b = orders_b.cache()
    timed(f"{label} - 1st count of bronze", lambda: (users_b.count(), orders_b.count()))
    timed(f"{label} - 2nd count of bronze", lambda: (users_b.count(), orders_b.count()))
    timed(f"{label} - build stg_orders", lambda: build_stg_orders(orders_b).count())

# ---------------------------------------------------------------------------
# Exercise 4: Iceberg time travel vs a Parquet directory
# ---------------------------------------------------------------------------
print("\n=== Exercise 4: Iceberg time travel vs Parquet ===")
spark.sql("CREATE NAMESPACE IF NOT EXISTS lab.silver")
stg_orders.writeTo("lab.silver.orders").createOrReplace()
snapshot_before = spark.sql(
    "SELECT snapshot_id FROM lab.silver.orders.snapshots ORDER BY committed_at DESC LIMIT 1"
).first()["snapshot_id"]

existing_user = stg_users.first()["user_id"]
new_order = spark.createDataFrame(
    [(
        "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
        existing_user,
        datetime.datetime(2020, 5, 1, 12, 0, tzinfo=datetime.timezone.utc),
        3,
        "croissant",
    )],
    schema=stg_orders.schema,
)

# Iceberg: append one order, then read the table as it was before
new_order.writeTo("lab.silver.orders").append()
print("Snapshots of lab.silver.orders:")
spark.sql("SELECT snapshot_id, operation, committed_at FROM lab.silver.orders.snapshots ORDER BY committed_at").show(truncate=False)
print("Iceberg - current version:", spark.table("lab.silver.orders").count(), "orders")
before = spark.sql(f"SELECT count(*) AS orders FROM lab.silver.orders VERSION AS OF {snapshot_before}").first()["orders"]
print(f"Iceberg - VERSION AS OF {snapshot_before}:", before, "orders")

# Parquet: same append on a plain directory
parquet_path = f"s3a://{BUCKET}/silver/spark/orders_exercise.parquet"
stg_orders.write.mode("overwrite").parquet(parquet_path)
print("Parquet - before append:", spark.read.parquet(parquet_path).count(), "orders")
new_order.write.mode("append").parquet(parquet_path)
print("Parquet - after append: ", spark.read.parquet(parquet_path).count(), "orders")
print("Parquet keeps no history: the directory only contains the current files, the previous version is lost.")

input("\nOpen the Storage tab of the Spark UI (port 4040), then press Enter to stop Spark...")
spark.stop()