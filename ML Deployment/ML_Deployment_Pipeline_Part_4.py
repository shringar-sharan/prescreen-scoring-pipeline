# Databricks notebook source
import numpy as np
import datetime as dt
import joblib

import pandas as pd
import re
from pyspark.sql.functions import col, lit
import pyspark.sql.functions as F

import mlflow
from pyspark.sql.functions import struct, col, when

folder_path = "/dbfs/mnt/science/Shringar/Databricks_Deployment/"

# COMMAND ----------

rrv9_features_path = folder_path + 'Features/Version_Agnostic_Features/rrv9_version_agnostic_features.csv'
rr_features_path = folder_path + 'Features/Version_Agnostic_Features/rrv10_version_agnostic_features.csv'
ar_features_path = folder_path + 'Features/Version_Agnostic_Features/arv7_version_agnostic_features.csv'
fr_features_path = folder_path + 'Features/Version_Agnostic_Features/frv7_version_agnostic_features.csv'
preUOL_features_path = folder_path + 'Features/Version_Agnostic_Features/preuolv1_version_agnostic_features.csv'

# COMMAND ----------

# rrv10_features_path = folder_path + 'Features/Version_Agnostic_Features/rrv10_version_agnostic_features.csv'
# rrv9_features_path = folder_path + 'Features/Version_Agnostic_Features/rrv9_version_agnostic_features.csv'

# rrv9_fts = pd.read_csv(rrv9_features_path, header=None)[0]
# rrv10_fts = pd.read_csv(rrv10_features_path, header=None)[0]

# all([a==b for a,b in zip(rrv9_fts, rrv10_fts)])

# COMMAND ----------

# rr_old_features_path = folder_path + 'Features/RR_v9_features.csv'
# ar_old_features_path = folder_path + "Features/ARv7_features_validation.csv"
# fr_old_features_path = folder_path + "Features/FRv7_features_validation.csv"

# COMMAND ----------

rr_fts = list(pd.read_csv(rr_features_path, header=None)[0])
ar_fts = list(pd.read_csv(ar_features_path, header=None)[0])
fr_fts = list(pd.read_csv(fr_features_path, header=None)[0])
preUOL_fts = list(pd.read_csv(preUOL_features_path, header=None)[0])

# COMMAND ----------

spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
from pyspark.sql.types import DateType, StringType, IntegerType
from pyspark.sql.functions import to_date, datediff, col, when, to_timestamp

def Part4_creating_reusable_features(df):
  
  df = (df.withColumn("CREDITASOFDATE", to_date(to_timestamp(col("creditAsOfDate").cast(StringType()), "yyyyMMdd")))
          .withColumn("LAST_ONSITE_APPLIED_DATE", to_date(to_timestamp(col("last_applied_date").cast(StringType()), "yyyy-MM-dd")))
          .withColumn("LAST_ONSITE_APPLIED_DATE", when(col("LAST_ONSITE_APPLIED_DATE").isNull(),'2010-01-01').otherwise(col("LAST_ONSITE_APPLIED_DATE")))
          .withColumn("months_since_last_onsite_application", (datediff(col("CREDITASOFDATE"), col("LAST_ONSITE_APPLIED_DATE"))/30).cast(IntegerType()))
          .withColumn("months_since_last_onsite_application", when(col("LAST_ONSITE_APPLIED_DATE") == '2010-01-01', 999).otherwise(col("months_since_last_onsite_application")))
          .withColumn("LAST_PARTNER_APPLIED_DATE", to_date(to_timestamp(col("last_partner_applied_date").cast(StringType()), "yyyy-MM-dd")))
          .withColumn("LAST_PARTNER_APPLIED_DATE", when(col("LAST_PARTNER_APPLIED_DATE").isNull(),'2010-01-01').otherwise(col("LAST_PARTNER_APPLIED_DATE")))
          .withColumn("months_since_last_partner_application", (datediff(col("CREDITASOFDATE"), col("LAST_PARTNER_APPLIED_DATE"))/30).cast(IntegerType()))
          .withColumn("months_since_last_partner_application", when(col("LAST_PARTNER_APPLIED_DATE") == '2010-01-01', 999).otherwise(col("months_since_last_partner_application")))
          .withColumn("months_since_last_application", when(col("months_since_last_partner_application") > col("months_since_last_onsite_application"), col("months_since_last_onsite_application")).otherwise(col("months_since_last_partner_application"))))

  df = (df.withColumn("PREVIOUS_COUNTER_5", when(col("previous6Tag1") == 'Y',1).otherwise(0) + 
                         when(col("previous6Tag2") == 'Y',1).otherwise(0) + 
                         when(col("previous6Tag3") == 'Y',1).otherwise(0) + 
                         when(col("previous6Tag4") == 'Y',1).otherwise(0) + 
                         when(col("previous6Tag5") == 'Y',1).otherwise(0))
           .withColumn("PREVIOUS_COUNTER_4", when(col("previous6Tag1") == 'Y',1).otherwise(0) + 
                         when(col("previous6Tag2") == 'Y',1).otherwise(0) + 
                         when(col("previous6Tag3") == 'Y',1).otherwise(0) + 
                         when(col("previous6Tag4") == 'Y',1).otherwise(0))
           .withColumn("PREVIOUS_COUNTER_3", when(col("previous6Tag1") == 'Y',1).otherwise(0) + 
                         when(col("previous6Tag2") == 'Y',1).otherwise(0) + 
                         when(col("previous6Tag3") == 'Y',1).otherwise(0))
           .withColumn("PREVIOUS_COUNTER_1", when(col("previous6Tag1") == 'Y',1).otherwise(0))
          )
  df = (df
           .withColumn("PREVIOUS_COUNTER_5", when(col("PREVIOUS_COUNTER_5") == 0, np.nan).otherwise(col("PREVIOUS_COUNTER_5")))
           .withColumn("PREVIOUS_COUNTER_4", when(col("PREVIOUS_COUNTER_4") == 0, np.nan).otherwise(col("PREVIOUS_COUNTER_4")))
           .withColumn("PREVIOUS_COUNTER_3", when(col("PREVIOUS_COUNTER_3") == 0, np.nan).otherwise(col("PREVIOUS_COUNTER_3"))) 
           .withColumn("PREVIOUS_COUNTER_1", when(col("PREVIOUS_COUNTER_1") == 0, np.nan).otherwise(col("PREVIOUS_COUNTER_1"))) 
          )
  df = (df
        .withColumn("W55_AGG911_Processed",  when((col("W55_AGG911").isNull())| (col("W55_AGG911") < 0), 0).otherwise(col("W55_AGG911")))
        .withColumn("W55_RVLR02_Processed",  when((col("W55_RVLR02").isNull())| (col("W55_RVLR02") < 0), 0).otherwise(col("W55_RVLR02")))
        .withColumn("W55_TRV18_Processed",  when((col("W55_TRV18").isNull())| (col("W55_TRV18") < 0), 0).otherwise(col("W55_TRV18")))
        .withColumn("W55_TRV06_Processed",  when((col("W55_TRV06").isNull())| (col("W55_TRV06") < 0), 0).otherwise(col("W55_TRV06")))
        .withColumn("V71_G205S_Processed",  when((col("V71_G205S").isNull())| (col("V71_G205S") < 0), 0).otherwise(col("V71_G205S")))
        .withColumn("W55_PAYMNT02_Processed",  when((col("W55_PAYMNT02").isNull())| (col("W55_PAYMNT02") < 0), 0).otherwise(col("W55_PAYMNT02")))
        .withColumn("W55_RVLR02_Processed",  when((df.W55_RVLR02.isNull())| (col("W55_RVLR02") < 0), 0).otherwise(col("W55_RVLR02")))
        .withColumn("CCAprEst",  (col("W49_ATTR08") / col("W49_ATTR07")) * 12 - 0.12)
        .withColumn("CCAprEst",  when(col("CCAprEst") > 1,    0   ).otherwise(col("CCAprEst")))
        .withColumn("CCAprEst",  when(col("CCAprEst") > 0.25, 0.25).otherwise(col("CCAprEst")))
        .withColumn("CCAprEst",  when(col("CCAprEst") < 0,    0   ).otherwise(col("CCAprEst")))   
       )
  df = (df
        .withColumn("CCAprEst",  F.when(df.CCAprEst.isNull(),     0   ).otherwise(col("CCAprEst")))  
       )
  df = (df
        .withColumn("W55_AGG911_Processed",  when(col("W55_AGG911_Processed") > 150, 150).otherwise(col("W55_AGG911_Processed")))
        .withColumn("W55_RVLR02_Processed",  when(col("W55_RVLR02_Processed") > 150, 150).otherwise(col("W55_RVLR02_Processed")))
        .withColumn("W55_TRV18_Processed",  when(col("W55_TRV18_Processed") > 12, 12).otherwise(col("W55_TRV18_Processed")))
        .withColumn("W55_TRV06_Processed",  when(col("W55_TRV06_Processed") > 12, 12).otherwise(col("W55_TRV06_Processed")))
        .withColumn("V71_G205S_Processed",  when(col("V71_G205S_Processed") > 10000, 10000).otherwise(col("V71_G205S_Processed")))
        .withColumn("MeanIncome_Processed",  when(col("MeanIncome") > 5e+05, 5e+05).otherwise(col("MeanIncome")))
        .withColumn("MeanIncome_Processed",  when(col("MeanIncome_Processed") < 1000, 1000).otherwise(col("MeanIncome_Processed")))
        
        .withColumn("no_student_trades",  when(F.col("V71_ST02S") == -1, 1).otherwise(0))
        .withColumn("W55_PAYMNT08_FRM",  when((df.W55_PAYMNT08.isNull())| (col("W55_PAYMNT08") < 0), 7.78911).otherwise(col("W55_PAYMNT08")))
        .withColumn("W55_PAYMNT08_FRM",  when(F.col("W55_PAYMNT08") == -1, 1.10877).otherwise(col("W55_PAYMNT08_FRM")))
        .withColumn("W55_PAYMNT08_FRM",  when(df.W55_PAYMNT08.isin(-2, -3), 4.58297).otherwise(col("W55_PAYMNT08_FRM")))
        .withColumn("W55_PAYMNT08_FRM",  when(F.col("W55_PAYMNT08") == -4, 33.99492).otherwise(col("W55_PAYMNT08_FRM")))
        
        
        .withColumn("MONTHS_SINCE_LAST_APPLICATION_ONSITE",  F.when(F.col("months_since_last_onsite_application") == 999, np.nan).otherwise(col("months_since_last_onsite_application")))
  
        .withColumn("CCAprEst_revolving",  (F.col("V71_REAP01") / F.col("V71_RE101S"))*12 - 0.12)
        .withColumn("CCAprEst_revolving",  F.when(col("CCAprEst_revolving") > 1,    0   ).otherwise(col("CCAprEst_revolving")))
        .withColumn("CCAprEst_revolving",  F.when(col("CCAprEst_revolving") > 0.25, 0.25).otherwise(col("CCAprEst_revolving")))
        .withColumn("CCAprEst_revolving",  F.when(col("CCAprEst_revolving") < 0,    0   ).otherwise(col("CCAprEst_revolving")))
       )
  
  df = (df
        .withColumn("CCAprEst_revolving",  F.when(col("CCAprEst_revolving").isNull(), 0).otherwise(col("CCAprEst_revolving")))  
       )
  df = (df
        .withColumn("state_label_5_FR", 
                    F.when((col("NCOA_ADDR_State").isin("WY","NH","ME","ND","AR", "HI","KY","DE","VT","SD", "WI","LA","UT","MT","IN", "SC","RI","FL","MI")), 0)
                     .when((col("NCOA_ADDR_State").isin("PA","TN","NC","NJ","WA", "OR","OH" )), 1)
                     .when((col("NCOA_ADDR_State").isin("IL","AL","ID","MD","CT", "GA","MN","WV","OK","MO", "VA")), 2)
                     .when((col("NCOA_ADDR_State").isin("CA","NM","KS","AZ","IA")), 3)
                     .when((col("NCOA_ADDR_State").isin("NY","CO","TX","AK","DC")), 4)
                     .otherwise(np.nan)))
  return df

# COMMAND ----------

def score_RR(df, rr_features_path = rr_features_path, model_uri=f"models:/DM_RR_v10/Production", output_col = 'RR_ResponseProb'):
  # Read RR_features
  features = list(pd.read_csv(rr_features_path, header=None)[0])
  
  # Loading model in production
  model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type='double')
  
  print("Scoring RR --> Done")
  
  return df.withColumn(output_col, model(struct(*map(col, features))))

# COMMAND ----------

def score_AR(df, ar_features_path = ar_features_path, model_uri=f"models:/DM_AR_v7/Production", output_col = 'AR_ApprovalProb'):
  # Read AR_features
  features = list(pd.read_csv(ar_features_path, header=None)[0])
  
  # Loading model in production
  model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type='double')
  
  print("Scoring AR --> Done")
  
  return df.withColumn(output_col, model(struct(*map(col, features))))

# COMMAND ----------

def score_FR(df, fr_features_path = fr_features_path, model_uri=f"models:/DM_FR_v7/Production", output_col = 'FR_FundingProb'):
  # Read FR_features
  features = list(pd.read_csv(fr_features_path, header=None)[0])
  
  # Loading model in production
  model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type='double')
  
  print("Scoring FR --> Done")
  
  return df.withColumn(output_col, model(struct(*map(col, features))))

# COMMAND ----------

# def score_preUOL(df, preUOL_features_path = preUOL_features_path, model_uri=f"models:/DM_preUOL_v1/Production", output_col = 'preUOLScore'):
#   # Read preUOL_features
#   features = list(pd.read_csv(preUOL_features_path, header=None)[0])
  
#   # Loading model in production
#   model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type='double')
  
#   return df.withColumn(output_col, model(struct(*map(col, features))))
