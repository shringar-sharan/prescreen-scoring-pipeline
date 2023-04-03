# Databricks notebook source
import pandas as pd
import numpy as np
import csv
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

import json, csv, os
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 200)
folder_path = "/mnt/dm_processed/cleaned/" 

# COMMAND ----------

interim = spark.read.csv("/mnt/dm_processed/cleaned/PSCamp124/J119565_D20220804_S481091_interimOutput_P001.dat.csv", header=True, inferSchema=True).limit(100000)

# COMMAND ----------

interim.display()

# COMMAND ----------

print('Interim file test sample shape:', interim.count(), len(interim.columns))

# COMMAND ----------

# MAGIC %md ## Preprocessing functions

# COMMAND ----------

# MAGIC %md ### Subset columns

# COMMAND ----------

def subset_variables_tiny(df):
  

# COMMAND ----------

# MAGIC %md ## Scoring functions

# COMMAND ----------

interim.write.format("delta").saveAsTable("default.interim")

# COMMAND ----------


