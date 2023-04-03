# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Adding DM Models to Mlflow Model Registry
# MAGIC 
# MAGIC Here are examples of using Mlflow Model Registry (https://www.mlflow.org/docs/latest/model-registry.html) to centralize our DM models. This allows them to be easily loaded into Python (pandas), PySpark, and R. 
# MAGIC 
# MAGIC Each model will end up with a URI like `models:/<model_name>/<model_version>` . 
# MAGIC 
# MAGIC You can then load that model and batch score in pyspark like this: 
# MAGIC 
# MAGIC ```
# MAGIC import mlflow.pyfunc
# MAGIC from pyspark.sql.functions import struct
# MAGIC 
# MAGIC varnames = ["feature1", "feature2", "feature3"]
# MAGIC 
# MAGIC predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri="models:/<model_name>/<model_version>")
# MAGIC df = df.withColumn("score", predict_udf(struct(*varnames)))
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Xgboost not install in R environment by default, this installs it
# MAGIC %r
# MAGIC install.packages("xgboost") 

# COMMAND ----------

import joblib

# COMMAND ----------

xgb = joblib.load("/dbfs/mnt/science/Shringar/Databricks_Deployment/Models/credit_model_v6.json")

# COMMAND ----------

# DBTITLE 1,Loading RDS files to get model objects (not needed if model objects are saved as standard binary files)
# MAGIC %r
# MAGIC library(xgboost)
# MAGIC # funding rate model v41
# MAGIC #xgb_model <- readRDS("/dbfs/mnt/science/Shringar/Databricks_Deployment/Models/credit_model_v6.rds")
# MAGIC xgb_model <- xgboost::xgb.load("/dbfs/mnt/science/Shringar/Databricks_Deployment/Models/credit_model_v6_resaving.json")
# MAGIC 
# MAGIC #readr::write_csv(as.data.frame(rrv9_vars), "/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/RR_v9_features.csv")

# COMMAND ----------

# MAGIC %r
# MAGIC xgb_model.feature_names

# COMMAND ----------

# MAGIC %r
# MAGIC xgb.save(xgb_model, "/dbfs/mnt/science/Shringar/Databricks_Deployment/Models/credit_model_v6_resaving.json")

# COMMAND ----------

import xgboost as xgb

# COMMAND ----------

# MAGIC %r
# MAGIC library(xgboost)
# MAGIC response_model_v6 <- xgboost::xgb.load("/dbfs/mnt/science/Chong/databricks/models/response_model_v6.model")
# MAGIC response_model_v7 <- xgboost::xgb.load("/dbfs/mnt/science/Chong/databricks/models/response_model_v7.model")
# MAGIC response_model_v8 <- xgboost::xgb.load("/dbfs/mnt/science/Chong/databricks/models/response_model_v8.model")
# MAGIC 
# MAGIC     

# COMMAND ----------

# MAGIC %fs ls /mnt/jason/dm_rework/PayoffDirectMail/data-raw/ResponseModel/v6

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC rm6_rds <- readRDS("/dbfs/mnt/jason/dm_rework/PayoffDirectMail/data-raw/ResponseModel/v6/response_model_v6.rds")
# MAGIC 
# MAGIC rm6 <- xgb.Booster.complete(rm6_rds)
# MAGIC 
# MAGIC xgb.save(rm6, "/dbfs/mnt/jason/dm_rework/response_model_v6.model")

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC test <- xgb.load("/dbfs/mnt/jason/dm_rework/response_model_v6.model")

# COMMAND ----------

import xgboost
# response rate model
response_model_v6 = xgboost.Booster(model_file ="/dbfs/mnt/jason/dm_rework/response_model_v6.model")

# COMMAND ----------

import xgboost
# response rate model
response_model_v6 = xgboost.Booster(model_file = "/dbfs/mnt/science/Chong/databricks/models/response_model_v6.model")

    

# COMMAND ----------

# add attribute to the object before registering
import pandas as pd
v41_names = pd.read_csv("/dbfs/mnt/jason/dm_rework/v41_names.csv")

varnames = v41_names.values.flatten().tolist()


#FM_v41.feature_names = varnames

# COMMAND ----------

# DBTITLE 1,Save the xgboost model object to a temporary location
# MAGIC %r
# MAGIC library(xgboost)
# MAGIC 
# MAGIC 
# MAGIC xgb.save(funding_model_v41, "/databricks/driver/funding_model_v41.xgboost")

# COMMAND ----------

# DBTITLE 1,Load the model in Python, then use Mlflow to register it

import mlflow.xgboost
import xgboost as xgb
import cloudpickle

# bst = xgb.Booster(model_file="/dbfs/mnt/jason/dm_rework/PayoffDirectMail/inst/extdata/IncomeModel/FinalModel.xgboost")
FM_v4 = xgb.Booster(model_file="/databricks/driver/funding_model_v41.xgboost") 

#optional, specify the version of xgboost needed for this model. Otherwise it will default to whichever one is installed right now
print("current version of xgboost installed is: {}".format(xgb.__version__))

conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(3.7),
      'pip',
      {
        'pip': [
          'mlflow',
          'xgboost==1.0.*',
          'cloudpickle=={}'.format(cloudpickle.__version__),
        ],
      },
    ],
    'name': 'xgb_env'
}

#log the model object and register it. Version increments every time you do this
mlflow.xgboost.log_model(xgb_model=FM_v4, artifact_path="FM_v4", registered_model_name="FM_v4", conda_env=conda_env)


# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Load the model object
import mlflow 

FM_v4 = mlflow.xgboost.load_model("models:/FM_v4/1")

FM_v4

# COMMAND ----------

FM_v4.feature_names = ["feature1", "feature2"]

# COMMAND ----------

# if it's possible to check the model features? - yes for RDS object but no for .xgboost_model
FM_v4

# COMMAND ----------

# load testing df

fname = "/mnt/science/Chong/databricks/temp_data/sample_3_after_all_models.csv" #validation set of raw data

#load the delta table (basically a parquet file)
df = spark.read.csv(fname, header=True, inferSchema=True)


# COMMAND ----------

# do scoring after loading the model object

import mlflow.pyfunc
from pyspark.sql.functions import struct


FM_v4_varnames = ["W55_AGG911_Processed", "W55_RVLR02_Processed", "W55_TRV18_Processed", "V71_G205S_Processed", "W55_PAYMNT02_Processed", "MeanIncome_Processed",  
            "GGENERC_gna001",  "CCAprEst",  "V71_AT34B", "V71_MT09S", "V71_BR31S", "V71_REAP01", "no_student_trades","W55_WALSRVS1", "V71_RE28S", "V71_INAP01", 
            "V71_BR01S", "W55_TRV14", "V71_G200S",  "V71_G201A", "V71_ST05S", "W49_AUC1002", "W49_AUP1003", "Model_v3_Score" ]

predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri="models:/FM_v4/1")
df = df.withColumn("FR_v4_python", predict_udf(struct(*FM_v4_varnames)))


# COMMAND ----------

test = df.select("FR_v4_python", "FundingProb_v41")

# test_R = df.select("FundingProb_v41")

display(test)

# COMMAND ----------

import numpy as np

test_pandas = test.toPandas()

np.allclose(test_pandas.FR_v4_python, test_pandas.FundingProb_v41, equal_nan=True)



# COMMAND ----------

# MAGIC %python
# MAGIC import mlflow
# MAGIC import xgboost as xgb
# MAGIC import cloudpickle
# MAGIC 
# MAGIC bst = xgb.Booster(model_file="/databricks/driver/funding_model_v41.xgboost")
# MAGIC 
# MAGIC 
# MAGIC conda_env = {
# MAGIC     'channels': ['defaults'],
# MAGIC     'dependencies': [
# MAGIC       'python={}'.format(3.7),
# MAGIC       'pip',
# MAGIC       {
# MAGIC         'pip': [
# MAGIC           'mlflow',
# MAGIC           'xgboost==1.0.*',
# MAGIC           'cloudpickle=={}'.format(cloudpickle.__version__),
# MAGIC         ],
# MAGIC       },
# MAGIC     ],
# MAGIC     'name': 'xgb_env'
# MAGIC }
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC mlflow.xgboost.log_model(bst, "model", registered_model_name="dm_funding_model")

# COMMAND ----------

# DBTITLE 1,I guessed this was the model object for PIEv4 - it wasn't in the PayoffDirectMail library
# MAGIC %fs cp /mnt/science/Chong/PIE/ec2_back_up/PIEv4_model/Turgut_PIE_v4/PIEv4.xgb /mnt/jason/dm_rework/PIEv4/PIEv4.xgb

# COMMAND ----------

bst4 = xgb.Booster(model_file="/dbfs/mnt/science/Chong/PIE/ec2_back_up/PIEv4_model/Turgut_PIE_v4/PIEv4.xgb")

mlflow.xgboost.log_model(xgb_model=bst4, artifact_path="PIEv4", registered_model_name="PIEv4")

# COMMAND ----------

# DBTITLE 1,Deleting a model from the Model Registry
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.delete_registered_model("FM_v4")


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Other models we need to load into the Model Registry

# COMMAND ----------

# MAGIC %r
# MAGIC load("/dbfs/mnt/jason/dm_rework/PayoffDirectMail/inst/extdata/OriginationFee/origination_fee.model")
# MAGIC 
# MAGIC origination_fee.model
# MAGIC 
# MAGIC 
# MAGIC # - Chong - rewrite the R function
# MAGIC #https://stash.int.payoff.com/projects/DS/repos/dm_scoring_scripts/browse/2020-12/003.1_score_dm_202012_PIEv3.R#168

# COMMAND ----------

# MAGIC %r
# MAGIC code_model_001 <- readRDS("/dbfs/mnt/science/Chong/0_EC2_backup/09_creative_model/v1/code_model_001.rds")
# MAGIC code_model_003 <- readRDS("/dbfs/mnt/science/Chong/0_EC2_backup/09_creative_model/v1/code_model_003.rds")
