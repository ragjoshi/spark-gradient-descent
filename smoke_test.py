from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("smoke-test") \
    .master("local[8]") \
    .getOrCreate()

sc = spark.sparkContext
rdd = sc.parallelize(range(1_000_000), numSlices=8)
print("Partitions:", rdd.getNumPartitions())
print("Sum via treeReduce:", rdd.treeReduce(lambda a, b: a + b))

spark.stop()