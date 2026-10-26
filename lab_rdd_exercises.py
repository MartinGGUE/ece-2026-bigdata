"""Lab RDD - exercises 1 to 4."""

import re
import time
from operator import add

from pyspark.sql import Row, SparkSession

spark = SparkSession.builder.appName("lab-rdd-exercises").master("local[*]").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

UUID_AT_START = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def parse_order(line):
    order_id, user_id, date, quantity, product = line.split(",")
    return {
        "order_id": order_id,
        "user_id": user_id,
        "date": date,
        "quantity": int(quantity),
        "product": product.strip().lower(),
    }


def load_orders():
    lines = sc.textFile("orders.csv")
    header = lines.first()
    return lines.filter(lambda line: line != header and line.strip() != "").map(parse_order)


orders_rdd = load_orders().cache()
users_records = sc.textFile("users.csv").filter(lambda line: UUID_AT_START.match(line) is not None)
users_kv = users_records.map(lambda line: (line.split(",")[0], line.split(",")[1]))

print("\n=== Exercise 1: aggregateByKey ===")
stats = orders_rdd.map(lambda o: (o["user_id"], o["quantity"])).aggregateByKey(
    (0, 0),                                     
    lambda acc, q: (acc[0] + 1, acc[1] + q),    
    lambda a, b: (a[0] + b[0], a[1] + b[1]),    
)
avg_quantity = stats.mapValues(lambda s: round(s[1] / s[0], 2))
for user_id, (order_count, total_quantity) in stats.take(5):
    print(f"{user_id}  orders={order_count}  total={total_quantity}  avg={total_quantity / order_count:.2f}")
print("Average quantity per order (first 5 users):", avg_quantity.take(5))

print("\n=== Exercise 2: leftOuterJoin with a fake user_id ===")
FAKE_USER_ID = "00000000-0000-0000-0000-000000000000"
fake_order = {
    "order_id": "ffffffff-ffff-ffff-ffff-ffffffffffff",
    "user_id": FAKE_USER_ID,
    "date": "2020-05-01 00:00:00+00:00",
    "quantity": 1,
    "product": "bread",
}
orders_with_fake = orders_rdd.union(sc.parallelize([fake_order]))
joined = orders_with_fake.map(lambda o: (o["user_id"], o)).leftOuterJoin(users_kv)
orphans = joined.filter(lambda kv: kv[1][1] is None)  
print("Orders without a matching user:", orphans.count())
print("Example:", orphans.take(1))

print("\n=== Exercise 3: RDD vs DataFrame ===")
user_quantity = orders_rdd.map(lambda o: (o["user_id"], o["quantity"])).reduceByKey(add)
orders_df = spark.createDataFrame(orders_rdd.map(lambda o: Row(**o)))
df_quantity = orders_df.groupBy("user_id").sum("quantity")
df_quantity.show(5, truncate=False)
print("--- DataFrame plan (.explain) ---")
df_quantity.explain()
print("--- RDD lineage (.toDebugString) ---")
print(user_quantity.toDebugString().decode())

print("\n=== Exercise 4: cache vs no cache ===")


def timed(label, func):
    start = time.perf_counter()
    result = func()
    print(f"{label:<40} {time.perf_counter() - start:.3f}s  (result={result})")


for name, rdd in [("with cache", load_orders().cache()), ("without cache", load_orders())]:
    per_user = rdd.map(lambda o: (o["user_id"], o["quantity"])).reduceByKey(add)
    timed(f"{name} - orders count (1st)", rdd.count)
    timed(f"{name} - orders count (2nd)", rdd.count)
    timed(f"{name} - user_quantity count", per_user.count)

input("\nOpen the Storage tab of the Spark UI (port 4040), then press Enter to stop Spark...")
spark.stop()