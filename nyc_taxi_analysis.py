from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType

spark = SparkSession.builder \
    .appName("NYC Taxi Trip Analysis") \
    .getOrCreate()
    
df = spark.read.csv("hdfs://localhost:9000/user/sanzhar/nyc_taxi_data/cleaned_output", header=False, inferSchema=True)

columns = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "pickup_longitude", "pickup_latitude", "RateCodeID", "store_and_fwd_flag",
    "dropoff_longitude", "dropoff_latitude", "payment_type", "fare_amount", "extra", "mta_tax",
    "tip_amount", "tolls_amount", "improvement_surcharge", "total_amount"
]
df = df.toDF(*columns)

df = df.withColumn("pickup_datetime", to_timestamp("tpep_pickup_datetime")) \
       .withColumn("dropoff_datetime", to_timestamp("tpep_dropoff_datetime"))

df = df.withColumn("pickup_latitude_rounded", round("pickup_latitude", 3)) \
       .withColumn("pickup_longitude_rounded", round("pickup_longitude", 3)) \
       .withColumn("dropoff_latitude_rounded", round("dropoff_latitude", 3)) \
       .withColumn("dropoff_longitude_rounded", round("dropoff_longitude", 3))

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

routes = df.groupBy("pickup_latitude_rounded", "pickup_longitude_rounded", "dropoff_latitude_rounded", "dropoff_longitude_rounded") \
    .count() \
    .orderBy(desc("count"))

top_routes = routes.limit(20)
top_routes.show() 

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

df = df.withColumn("trip_duration", (unix_timestamp("dropoff_datetime") - unix_timestamp("pickup_datetime")) / 3600) 
df = df.withColumn("speed_kmh", col("trip_distance") / col("trip_duration"))
congestion_threshold = 20

congested_routes = df.filter(col("speed_kmh") < congestion_threshold)

congested_routes_2 = congested_routes.withColumn("hour", hour("pickup_datetime"))


#---------------------------------------------------------------------------------------------------------------------------------------------------------------

congested_routes = congested_routes.select("pickup_latitude_rounded", "pickup_longitude_rounded", "dropoff_latitude_rounded", "dropoff_longitude_rounded", 
                                          "pickup_datetime", "dropoff_datetime", "speed_kmh") \
    .orderBy("pickup_datetime")

congested_routes.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

congestion_by_hour = congested_routes.groupBy(hour("pickup_datetime").alias("hour")) \
    .agg(
        count("*").alias("num_congested_trips"),
        round(avg("speed_kmh"), 2).alias("avg_speed")
    ) \
    .orderBy("hour")

congestion_by_hour.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

repeated_congested_routes = congested_routes.groupBy("pickup_latitude_rounded", "pickup_longitude_rounded", "dropoff_latitude_rounded", "dropoff_longitude_rounded") \
    .agg(
        count("*").alias("num_repeated_congested_trips")
    ) \
    .filter(col("num_repeated_congested_trips") > 5)

repeated_congested_routes = repeated_congested_routes.orderBy(desc("num_repeated_congested_trips"))
repeated_congested_routes.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

hourly_stats = df.withColumn("hour", hour("pickup_datetime")) \
    .groupBy("hour") \
    .agg(
        round(avg("trip_distance"), 2).alias("avg_distance"),
        count("*").alias("num_trips")
    ) \
    .orderBy("hour")

hourly_stats.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

payment_type = df.groupBy("payment_type") \
  .count() \
  .orderBy(desc("count"))

payment_type.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

avr_total_amount = df.withColumn("day_of_week", date_format("pickup_datetime", "E")) \
  .groupBy("day_of_week") \
  .agg(
      round(avg("total_amount"), 2).alias("avg_total_amount"),
      count("*").alias("num_trips")
  ) \
  .orderBy("day_of_week")

avr_total_amount.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

passenger_stats = df.groupBy("passenger_count") \
  .count() \
  .orderBy("passenger_count")

passenger_stats.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

congested_routes_grouped = congested_routes_2.groupBy("hour", "pickup_latitude_rounded", "pickup_longitude_rounded", 
                                                    "dropoff_latitude_rounded", "dropoff_longitude_rounded") \
    .agg(
        count("*").alias("num_congested_trips"),
        round(avg("speed_kmh"), 2).alias("avg_speed")
    ) \
    .filter(col("num_congested_trips") > 5)

congested_routes_grouped = congested_routes_grouped.orderBy(desc("num_congested_trips"), "hour")


congested_routes_grouped.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

hourly_stats.write.csv("hdfs://localhost:9000/user/sanzhar/final/hourly_stats", header=True)
payment_type.write.csv("hdfs://localhost:9000/user/sanzhar/final/payment_type_count", header=True)
top_routes.write.csv("hdfs://localhost:9000/user/sanzhar/final/top_routes", header=True)
avr_total_amount.write.csv("hdfs://localhost:9000/user/sanzhar/final/avr_total_amount", header=True)
passenger_stats.write.csv("hdfs://localhost:9000/user/sanzhar/final/passenger_count", header=True)
congestion_by_hour.write.csv("hdfs://localhost:9000/user/sanzhar/final/congestion_by_hour", header=True)
repeated_congested_routes.write.csv("hdfs://localhost:9000/user/sanzhar/final/repeated_congested_routes", header=True)
congested_routes_grouped.write.csv("hdfs://localhost:9000/user/sanzhar/final/congested_routes_grouped", header=True)

spark.stop()

