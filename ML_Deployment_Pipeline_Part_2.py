# Databricks notebook source
import numpy as np
import datetime as dt
import joblib

import pandas as pd
import re
from pyspark.sql.functions import col, lit
import pyspark.sql.functions as F

import mlflow
from pyspark.sql.functions import struct, col

folder_path = "/dbfs/mnt/science/Shringar/Databricks_Deployment/"

# COMMAND ----------

pie_features_path = folder_path + 'Features/piev5_features.csv'
lae_features_path = folder_path + 'Features/Version_Agnostic_Features/laev3_version_agnostic_features.csv'

# COMMAND ----------

pie_features = list(pd.read_csv(pie_features_path, header=None)[0])
lae_features = list(pd.read_csv(lae_features_path, header=None)[0])

# COMMAND ----------

hpe_features = ['V71_G208S','MedianRental','PIE_Income_Estimation','V71_AT36S','V71_G208S','V71_BC03S','W55_TRV14','V71_RT35S','V71_MT01S','W49_ATTR09']

# COMMAND ----------

def score_PIE(df, pie_features_path = pie_features_path, model_uri=f"models:/DM_PIE_v5/Production"):
  # Read PIE_features
  features = list(pd.read_csv(pie_features_path, header=None)[0])
  
  # Loading model in production
  model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type='double')
  
  print("Scoring PIE --> Done")
  
  return df.withColumn('PIE_Income_Estimation', model(struct(*map(col, features))))

# COMMAND ----------

def score_HPE(df, model_uri=f"models:/DM_HPE_v2/Production"):
  # Loading model in production
  model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type='double')
  
  print("Scoring HPE --> Done")
  
  return df.withColumn('HPE_Housing_Payment_Estimation', model(struct(*map(col, df.columns))))

# COMMAND ----------

def score_LAE(df, lae_features_path = lae_features_path, model_uri=f"models:/DM_LAE_v3/Production"):
  df = df.withColumn("W49_AUC1002",col("W49_AUC1002").cast('int'))\
         .withColumn("lastreportedhousingpayment",col("lastreportedhousingpayment").cast('int'))
  
  # Read LAE_features
  features = list(pd.read_csv(lae_features_path, header=None)[0])
  
  # Loading model in production
  model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type='double')
  
  print("Scoring LAE --> Done")
  
  return df.withColumn('LAE_Loan_Amount_Estimation', model(struct(*map(col, features))))

# COMMAND ----------

def eligibility_bestcase_income(df, output_col = "bestcase_PIE_Income_Estimation_Or_Reported_Income_Blend"):
  df = df.withColumn(output_col, F.greatest(col("PIE_Income_Estimation"), col("lastreportedincome")))
  return df
