# Databricks notebook source
# https://github.com/Pay-Baymax/DataEngineerChallenge

# # DataEngineerChallenge

# This is an interview challenge for PayPay. Please feel free to fork. Pull Requests will be ignored.
# The challenge is to make make analytical observations about the data using the distributed tools below.

# ## Processing & Analytical goals:
# 1. Sessionize the web log by IP. Sessionize = aggregrate all page hits by visitor/IP during a session.
#     https://en.wikipedia.org/wiki/Session_(web_analytics)
# 2. Determine the average session time
# 3. Determine unique URL visits per session. To clarify, count a hit to a unique URL only once per session.
# 4. Find the most engaged users, ie the IPs with the longest session times

# ## Additional questions for Machine Learning Engineer (MLE) candidates:
# 1. Predict the expected load (requests/second) in the next minute
# 2. Predict the session length for a given IP
# 3. Predict the number of unique URL visits by a given IP

# ## Tools allowed (in no particular order):
# - Spark (any language, but prefer Scala or Java)
# - Pig
# - MapReduce (Hadoop 2.x only)
# - Flink
# - Cascading, Cascalog, or Scalding

# If you need Hadoop, we suggest 
# HDP Sandbox:
# http://hortonworks.com/hdp/downloads/
# or 
# CDH QuickStart VM:
# http://www.cloudera.com/content/cloudera/en/downloads.html


# ### Additional notes:
# - You are allowed to use whatever libraries/parsers/solutions you can find provided you can explain the functions you are implementing in detail.
# - IP addresses do not guarantee distinct users, but this is the limitation of the data. As a bonus, consider what additional data would help make better analytical conclusions
# - For this dataset, complete the sessionization by time window rather than navigation. Feel free to determine the best session window time on your own, or start with 15 minutes.
# - The log file was taken from an AWS Elastic Load Balancer:
# http://docs.aws.amazon.com/ElasticLoadBalancing/latest/DeveloperGuide/access-log-collection.html#access-log-entry-format

# ## How to complete this challenge:
# A. Fork this repo in github
# B. Complete the processing and analytics as defined first to the best of your ability with the time provided.
# C. Place notes in your code to help with clarity where appropriate. Make it readable enough to present to the PayPay interview team.
# D. Complete your work in your own github repo and send the results to us and/or present them during your interview.
# ## What are we looking for? What does this prove?

# We want to see how you handle:
# - New technologies and frameworks
# - Messy (ie real) data
# - Understanding data transformation
# This is not a pass or fail test, we want to hear about your challenges and your successes with this particular problem.

# COMMAND ----------

#---------------------------------------
#   PGM     : web log analysis
#   author  : min
#   create  : 2019/04/20
#   lang    : pyspark
#   func    : analyze the elb web log
#   history : 2019/04/20 new @min
#---------------------------------------

"""
Analysis of aws elb web log.
Use:
  - databricks spark env
  - databricks notebook
  - pyspark
Flow:
  - Parse elb web logs.
  - Sessionize.
  - Analyze session time.
  - Count unique requests.
  - Find longest session times.
"""

from pyspark.sql import SparkSession

from pyspark.sql.window import Window
from pyspark.sql.functions import lag, max, sum, mean
from pyspark.sql.functions import col, when, count, countDistinct
from pyspark.sql.functions import split, concat_ws

from pyspark.sql.types import StructField, StructType
from pyspark.sql.types import StringType, FloatType


# COMMAND ----------

def parse_logs(spark, logFile, numPartitions=10):
    """define schema from aws elb
    aws elb web Log format
    https://docs.aws.amazon.com/elasticloadbalancing/latest/classic/access-log-collection.html

    datatype
    https://spark.apache.org/docs/latest/sql-reference.html
    StructField(fieldName, StringType, nullable = true)"""
    log_schema = StructType([
        StructField("timestamp", StringType(), False),
        StructField("elb", StringType(), False),
        StructField("client:port", StringType(), False),
        StructField("backend:port", StringType(), False),
        StructField("request_processing_time", StringType(), False),
        StructField("backend_processing_time", StringType(), False),
        StructField("response_processing_time", StringType(), False),
        StructField("elb_status_code", StringType(), False),
        StructField("backend_status_code", StringType(), False),
        StructField("received_bytes", StringType(), False),
        StructField("sent_bytes", StringType(), False),
        StructField("request", StringType(), False),
        StructField("user_agent", StringType(), False),
        StructField("ssl_cipher", StringType(), False),
        StructField("ssl_protocol", StringType(), False)])
    #
    # load elb log data
    # repatitions and cache the df
    init_df = spark.read.csv(logFile, schema=log_schema, sep=" ").repartition(numPartitions).cache()
    # print (init_df.rdd.getNumPartitions())
    # 
    # etl access data
    split_clinet = split(init_df["client:port"], ":")
    split_backend = split(init_df["backend:port"], ":")
    split_request = split(init_df["request"], " ")
    # 
    df = init_df.withColumn("client_ip", split_clinet.getItem(0)) \
                .withColumn("client_port", split_clinet.getItem(1)) \
                .withColumn("backend_ip", split_backend.getItem(0)) \
                .withColumn("backend_port", split_backend.getItem(1)) \
                .withColumn("request_action", split_request.getItem(0)) \
                .withColumn("request_url", split_request.getItem(1)) \
                .withColumn("request_protocol", split_request.getItem(2)) \
                .withColumn("curr_timestamp", col("timestamp").cast("timestamp")) \
                .drop("client:port","backend:port","request").cache()
    # df.show(5)
    return df

# COMMAND ----------

def time_diff(start, end):
    """
    get diff between start and end
    """
    try:
        diff_secs = (end - start).total_seconds()
    except:
        diff_secs = 0
    return diff_secs
get_time_diff = udf(time_diff, FloatType())


def Sessionize(df_logs, session_time=15):
    """
    Sessionize the web log by IP.
    Sessionize = aggregrate all page hits by visitor/IP during a session.
    https://en.wikipedia.org/wiki/Session_(web_analytics)
    """
    session_time_secs = session_time * 60
    window_func_ip = Window.partitionBy("client_ip").orderBy("curr_timestamp")
    df = df_logs.withColumn("prev_timestamp", 
                            lag(col("curr_timestamp")).over(window_func_ip)) \
                .withColumn("session_lasts", 
                            get_time_diff(col("prev_timestamp"), col("curr_timestamp"))) \
                .withColumn("new_session_flag", 
                            when((col("session_lasts") > session_time_secs), 1).otherwise(0)) \
                .withColumn("count_session", 
                            sum(col("new_session_flag")).over(window_func_ip)) \
                .withColumn("ip_session_count", 
                            concat_ws("_", col("client_ip"), col("count_session")))
    
    df_ip_session = df.select(["ip_session_count", "client_ip", "request_url",
                               "prev_timestamp", "curr_timestamp", 
                               "session_lasts", "new_session_flag", "count_session"])
    return df_ip_session

# COMMAND ----------

def analyze_session_time(df_ip_session, save_res_to_csv=False):
    """
    Determine the average session time.
    """
    window_func_session = Window.partitionBy("ip_session_count").orderBy("curr_timestamp")
    df_session = df_ip_session.withColumn("prev_timestamp_session", 
                              lag(df_ip_session["curr_timestamp"]).over(window_func_session)) \
                  .withColumn("current_session_lasts", 
                              get_time_diff(col("prev_timestamp_session"), col("curr_timestamp")))
    #
    df_session_total = df_session.groupby("ip_session_count").agg(
            sum("current_session_lasts").alias("total_session_time")).cache()
    # Average session time.
    df_session_avg = df_session_total.select([mean("total_session_time").alias("avg_session_time")]).cache()
    if save_res_to_csv:
        df_session_avg.coalesce(1).write.csv(path="df_session_avg.csv", header=True, sep=",", mode="overwrite")
    df_session_avg.show()
    #
    return df_session

# COMMAND ----------

def count_unique_request(df_session):
    """
    Determine unique URL visits per session.
    To clarify, count a hit to a unique URL only once per session.
    """
    df_unique_url = df_session.groupby("ip_session_count").agg(
        countDistinct("request_url").alias("count_unique_requests"))
    df_unique_url.show()
    return df_unique_url

def get_longest_session_time(df_session):
    """
    get the longest session time
    """
    df_ip_time = df_session.groupby("client_ip").agg(
                sum("session_lasts").alias("session_time_all"),
                count("client_ip").alias("num_sessions"),
                max("session_lasts").alias("session_lasts_max")) \
          .withColumn("avg_session_time", col("session_time_all") / col("num_sessions")) \
          .orderBy(col("avg_session_time"), ascending=False)
    #
    df_ip_time.show()
    return df_ip_time

# COMMAND ----------

def exec_analysis(spark, logFile, numPartitions=10, session_time=15, save_res_to_csv=False):
    """
    Find the most engaged users, ie the IPs with the longest session times
    execute the analysis.
    """
    # parese the logs by regex -> slow
    # parse the logs
    df_logs = parse_logs(spark, logFile, numPartitions).repartition(numPartitions).cache()
    df_ip_session = Sessionize(df_logs, session_time).repartition(numPartitions).cache()
    df_session = analyze_session_time(df_ip_session, save_res_to_csv).repartition(numPartitions).cache()
    df_unique_url = count_unique_request(df_session).cache()
    df_ip_time = get_longest_session_time(df_session).cache()
    if save_res_to_csv:
        df_session.coalesce(1).write.csv(path="df_session.csv", header=True, sep=",", mode="overwrite")
        df_unique_url.coalesce(1).write.csv(path="df_unique_url.csv", header=True, sep=",", mode="overwrite")
        df_ip_time.coalesce(1).write.csv(path="df_ip_time.csv", header=True, sep=",", mode="overwrite")


# COMMAND ----------

##############main######################
if __name__ == "__main__":
    if "spark" not in dir():
      spark = SparkSession.builder \
        .appName("web_log_analysis") \
        .getOrCreate()
    #
    logFile = "/FileStore/tables/2015_07_22_mktplace_shop_web_log_sample_log-214a9.gz"
    numPartitions = 10
    session_time = 15
    save_res_to_csv = True
    #
    exec_analysis(spark, logFile, numPartitions, session_time, save_res_to_csv)
