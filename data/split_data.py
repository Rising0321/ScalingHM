from pyspark import SparkConf, SparkContext
from pyspark.sql import functions as fn
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
import time
import pyarrow.parquet as pq
import os
import math
from pyspark.sql.window import Window
import time
import datetime

os.environ["JAVA_HOME"] = "/home/wangb/zhangrx/TrainIdentification/data/JAVA8/jdk1.8.0_202"

# SparkContext.setSystemProperty("spark.driver.maxResultSize","1g")
spark = SparkSession.builder \
    .config("spark.excutor.memory", "300g") \
    .config("spark.driver.memory", "300g") \
    .config("spark.excutor.cores", 40) \
    .appName("Read Parquet") \
    .getOrCreate()

year = 2023
month = 3
day = 1

now_date = year * 10000 + month * 100 + day
nex_date = year * 10000 + month * 100 + day + 10

date_time = datetime.datetime(year, month, day)
now_unix = int(time.mktime(date_time.timetuple()))
date_time = datetime.datetime(year, month, day + 10)
nex_unix = int(time.mktime(date_time.timetuple()))

time_delta = 15 * 60

now_dir = 0


def change_pemu(a, b):
    result = np.empty_like(a)
    result[b] = a
    return result


def calculate_static(iteratrors):
    res = []
    res_id = []

    for now__ in iteratrors:
        # print(now__)
        now_ = [now__[0], change_pemu(list(now__[1]), list(now__[6])),
                change_pemu(list(now__[2]), list(now__[6])),
                change_pemu(list(now__[3]), list(now__[6])),
                change_pemu(list(now__[4]), list(now__[6])),
                change_pemu(list(now__[5]), list(now__[6]))]

        cnt = 0
        now = []
        pre = []

        max_len = len(now_[1])

        for i in range(max_len):

            x = [now_[0], now_[1][i], now_[2][i], now_[3][i], now_[4][i], now_[5][i]]

            if x[2] < 1 or x[3] < 1:
                continue

            if len(pre) == 0:
                lef = (now_unix - now_unix - 1) // time_delta
                rig = (x[5] - 1 - now_unix) // time_delta
                if rig - lef >= 1:
                    pre = x
                    cnt += int(rig - lef)
                    now.extend([[x[2], x[3]]] * int(rig - lef))


            else:
                lef = (pre[5] - now_unix - 1) // time_delta
                rig = (x[5] - 1 - now_unix) // time_delta
                if rig - lef >= 1:
                    pre = x
                    cnt += int(rig - lef)
                    now.extend([[x[2], x[3]]] * int(rig - lef))

        if len(pre) != 0:
            lef = (pre[5] - now_unix - 1) // time_delta
            rig = (nex_unix - 1 - now_unix) // time_delta
            if rig - lef >= 1:
                cnt += int(rig - lef)
                now.extend([[pre[2], pre[3]]] * int(rig - lef))
            res.append(now)
            res_id.append(now_[0])
            assert len(now) == 960, "error %d %d" % (len(now), cnt)

    try:
        # print(os.getpid())
        # print(len(res))
        now_id = os.getpid()
        while True:
            path = ("/home/wangb/zhangrx/TrajGPTData/%d/{}_data.npy" % now_dir).format(now_id)
            if os.path.exists(path):
                now_id += 20
            else:
                break

        np.save(("/home/wangb/zhangrx/TrajGPTData/%d/{}_data.npy" % now_dir).format(now_id), res)
        np.save(("/home/wangb/zhangrx/TrajGPTData/%d/{}_uid.npy" % now_dir).format(now_id), res_id)
    except Exception as e:
        print(e)


def output_data(full_data):
    window_spec = Window.partitionBy("UID").orderBy("procedureStartTime")

    full_data = full_data.withColumn(
        "rownum",
        fn.row_number().over(window_spec) - 1
    )

    full_data = full_data.groupby("UID").agg(
        fn.collect_list("CID").alias("CID_list"),
        fn.collect_list("longitude").alias("longitude_list"),
        fn.collect_list("latitude").alias("latitude_list"),
        fn.collect_list("procedureStartTime").alias("procedureStartTime_list"),
        fn.collect_list("procedureEndTime").alias("procedureEndTime_list"),
        fn.collect_list("rownum").alias("rownum_list")
    )
    print("-----------------------------------------")
    full_data.foreachPartition(calculate_static)


from tqdm import tqdm

now_st = 0
now_len = 100000
for sub_data in tqdm(range(10)):
    os.makedirs("/home/wangb/zhangrx/TrajGPTData/%d" % sub_data, exist_ok=True)
    now_dir = sub_data
    sum_data = None
    for i in range(1, 11):
        data_dir = "/home/wangb/zhangrx/TrainIdentification/data/parquet/202303%02d*" % i
        full_data = spark.read.parquet(data_dir)
        full_data = full_data.filter((full_data["UID"] < now_st + now_len) & (now_st <= full_data["UID"]))
        print(full_data.count())
        if sum_data == None:
            sum_data = full_data
        else:
            sum_data = sum_data.union(full_data)
    output_data(sum_data)
    now_st += now_len
