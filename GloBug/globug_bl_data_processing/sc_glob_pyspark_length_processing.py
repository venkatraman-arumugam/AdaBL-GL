import pandas as pd
import numpy as np
from pathlib import Path
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import array, lit, col, rand, rank
import pyspark
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql.types import *


# sc = pyspark.SparkContext('local[*]')
spark = SparkSession \
    .builder \
    .appName("BL") \
    .config("spark.some.config.option", "") \
    .getOrCreate()
sc_spark_df = spark.read.option("header",True).option("escape","\"").csv("/home/varumuga/scratch/thesis/replication/bench_bl_dataset/data/allSourceCodesProcessedLength")
# all_bug_df_spark_processed_df = spark.read.option('header', True).option('inferSchema', True).option('delimiter', ',').csv("/home/varumuga/scratch/thesis/replication/bench_bl_dataset/data/allSourceCodesProcessed")
br_pd_schema = StructType([
StructField("bug_id", StringType(), True),
StructField("cid", IntegerType(), True),
StructField("match", StringType(), True),
StructField("commit", StringType(), True),
StructField("project_name", StringType(), True)
])
bug_id2cid = pd.read_csv("cid_to_bug_id.csv")
br_pandas_df = spark.createDataFrame(bug_id2cid, br_pd_schema)

combined_df = sc_spark_df.join(br_pandas_df, (sc_spark_df.cid == br_pandas_df.cid) & (sc_spark_df.project_name == br_pandas_df.project_name)).select(sc_spark_df["*"], br_pandas_df["bug_id"], br_pandas_df["commit"], br_pandas_df["match"])

combined_df.write.options(header=True).option('inferSchema', True).option('delimiter', ',').csv("/home/varumuga/scratch/thesis/replication/bench_bl_dataset/data/all_source_codes_processed/")

# def calculate_len(sc_compressed):
#     sc_code = zlib.decompress(bytes.fromhex(sc_compressed)).decode()
#     return len(sc_code.split("-|-"))
# udf_sc_size = udf(lambda x: calculate_len(x), IntegerType()) # if the function returns an int
# df = all_bug_df_spark_processed_df.withColumn("size", udf_sc_size(col("file_content_processed"))) #"_3" being the column name of the column you want to consider
# df.show()

# df.write.option("header",True).option('delimiter', ',').csv("/home/varumuga/scratch/thesis/replication/bench_bl_dataset/data/allSourceCodesProcessedLength")