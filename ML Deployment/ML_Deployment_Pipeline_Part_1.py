# Databricks notebook source
import numpy as np
import datetime as dt
import joblib

import pandas as pd
import re
from pyspark.sql.functions import col, lit
import pyspark.sql.functions as F

folder_path = "/dbfs/mnt/science/Shringar/Databricks_Deployment/"

# COMMAND ----------

dm_name_mapping_path = folder_path + "dm_name_mapping.csv"

# COMMAND ----------

def load_dm_file(input_path, rename=True):
  """input_path: The S3 folder path where cleaned TU files are put by the Data Engg team. TU provides us with a raw file which is cleaned and broken up into 20 CSV files"""
  df = spark.read.csv(input_path, header=True, inferSchema=True)
  
  if rename:
    df = rename_dm_variables(df)
  
  print("Loading all files --> Done")
  return df

# COMMAND ----------

# dm_name_mapping = pd.read_csv("/dbfs/mnt/science/RServerSetup/dm_name_mapping.csv")
# dm_name_mapping

# COMMAND ----------

def rename_dm_variables(df):
    # Name mapping for renaming
    dm_name_mapping = pd.read_csv(dm_name_mapping_path)
    
    # Creating a dictionary with {old_name: new_name}
    mapping = dict(zip(dm_name_mapping['raw_lower'].str.strip(), dm_name_mapping['final'].str.strip())) #create a dictionary with {old_name: new_name}
    
    #generate a list of new names, and remove any spaces
    new_names = [re.sub("\s+", "", mapping[col.lower()]) if col.lower() in mapping else col.lower() for col in df.columns]

    df = df.toDF(*new_names)
    print("Renaming variables --> Done")
    return df

# COMMAND ----------

def create_new_variables(df):
  
  # Renaming RECENT_TAG
  df = df.withColumnRenamed("RECENT_TAG", "eligible_last_month")
  df = df.withColumn("RECENT_TAG", col("previous6Tag1"))
  
  df = df.withColumn("UnsecuredSummaryBalanceTUVars1", col("V71_RE101S") + col("V71_IN101S") - col("V71_HR101S") - col("V71_AU101S") - col("V71_ST101S"))
  df = df.withColumn("UnsecuredSummaryBalanceCustomVars", col("epay01attr25") + col("epay01attr19"))
  df = df.withColumn("UnsecuredSummaryBalanceTUVars2", col("V71_US101S") + col("V71_BC101S"))
  df = df.withColumn("UnsecuredSummayBalanceBlend", F.greatest(col("UnsecuredSummaryBalanceTUVars1"), col("UnsecuredSummaryBalanceCustomVars")))
  
  # Consider removing if variables not being used
  df = (df
        .withColumn('N05_AT36', lit(None).cast("double"))
        .withColumn('N05_BR28', lit(None).cast("double"))
        .withColumn('N05_BC98', lit(None).cast("double"))
        .withColumn('N05_G046', lit(None).cast("double"))
       )
  
  print("Creating new variables --> Done")
  return df

# COMMAND ----------

def append_external_data(df,
                         zipcode_col = 'NCOA_ADDR_ZipCode',
                         external_data_path='/mnt/science/jason_azureblob/dm_rework/', 
                         pie_external_data_path='/mnt/science/Shringar/Databricks_Deployment/Features/piev5_external_data_modeling.csv'):
  
  pop_density = spark.read.csv(external_data_path + "population_density_append.csv", header=True, inferSchema=True)
  income_distribution = spark.read.csv(external_data_path + "income_distribution_append.csv", header=True, inferSchema=True)
  zillow = spark.read.csv(external_data_path + "zillow_append.csv", header=True, inferSchema=True).drop("median_rental")
  income = spark.read.csv(external_data_path + "income_append.csv", header=True, inferSchema=True).drop("size", "mean_income")
  irs_2015_variables_no_missing_zips = spark.read.csv(external_data_path + "irs_2015_variables_no_missing_zips.csv", header=True, inferSchema=True)\
                                        .withColumnRenamed("ZIPCODE", "zipcode")\
                                        .select("zipcode", "mean_agi", "mean_advance_prem_credit", "mean_net_inv_income_tax",
                                                                       "mean_state_sales_tax", "prop_refund_edu_credit", "prop_single_returns",
                                                                       "mean_res_energy_credit", "prop_nr_edu_credit")
  
  # Appending PIE external data
  pie_ext_data = spark.read.csv(pie_external_data_path, header=True, inferSchema=True)

  df = df.withColumnRenamed(zipcode_col, "zipcode")\
        .join(pie_ext_data, on="zipcode", how="left")\
        .join(pop_density, on="zipcode", how="left")\
        .join(income_distribution, on="zipcode", how="left")\
        .join(zillow, on="zipcode", how="left")\
        .join(income, on="zipcode", how="left")\
        .join(irs_2015_variables_no_missing_zips, on="zipcode", how="left")\
        .withColumnRenamed("zipcode", zipcode_col)
        
  print("Appending external data --> Done")
  
  return df
