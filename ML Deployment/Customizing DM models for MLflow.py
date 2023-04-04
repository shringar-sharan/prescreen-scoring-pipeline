# Databricks notebook source
# MAGIC %md ### Import libraries

# COMMAND ----------

import mlflow.pyfunc
import math
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
#import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, udf, when, exp, pow
folder_path = "/dbfs/mnt/science/Shringar/Databricks_Deployment/"

# COMMAND ----------

# MAGIC %md ### Defining naming conventions
# MAGIC * While logging model, log model as modelName_modelVersion_pyfunc
# MAGIC * While moving model to production, rename model to DM_modelName_modelVersion

# COMMAND ----------

# MAGIC %md ### Sample data to test model inputs, outputs

# COMMAND ----------

df = pd.read_csv(folder_path + 'sample_126_scored.csv', dtype={'NCOA_ADDR_ZipCode':'string'}).rename(columns={'NCOA_ADDR_ZipCode':'zipcode'})
external_data = pd.read_csv(folder_path + 'Features/piev5_external_data_modeling.csv', dtype={'zipcode':'string'})
df['zipcode'] = df['zipcode'].str.zfill(5)
external_data['zipcode'] = external_data['zipcode'].str.zfill(5)
df = df.merge(external_data, how='left', on='zipcode')
df.head()

# COMMAND ----------

df.to_csv(folder_path + 'sample_126_w_ext_data_scored.csv', index=False)

# COMMAND ----------

[col for col in df.columns if 'PH' in col]

# COMMAND ----------

# df2 = spark.createDataFrame(df)
# df2.display()
df2 = df2.withColumnRenamed('PIE_v5', 'PIE_Income_Estimation')

# COMMAND ----------

# MAGIC %md ### HPE_v2

# COMMAND ----------

# Define the model class
class HousingPaymentEstimator(mlflow.pyfunc.PythonModel):

  def __init__(self, income_var = 'PIE_Income_Estimation', median_rental_var = "MedianRental"):
    self.income_var = income_var
    self.median_rental_var = median_rental_var
    super().__init__()

  def predict(self, context, df):
    income_var = self.income_var
    median_rental_var = self.median_rental_var
    
    # Pandas version
    df['V71_G208S'] = np.where(df['V71_G208S'].isnull(), -5, df['V71_G208S'])
    df['rent'] = df[median_rental_var] * np.exp(df[income_var] * (5.017e-06) + df["V71_AT36S"] * (-0.0001028) + df["V71_G208S"] * (-0.001778) + df["V71_BC03S"] * (-0.01558) + df["W55_TRV14"] * (0.007818) + df["V71_RT35S"] * (3.578e-05) - 0.897)
    df['rent'] = np.where(df['rent'] <= 10000, np.where(df['rent'] < 250, 250, df['rent']), 10000)
    df['V71_MT01S_t'] = np.where(df['V71_MT01S'].isnull(), 0, df['V71_MT01S'])
    df['W49_ATTR09_t'] = np.where(df['W49_ATTR09'].isnull(), 0, df['W49_ATTR09'])
    df['mortgage'] = df[income_var] * (0.003352) + df["V71_MT01S_t"] * (17.987879) + df["W49_ATTR09_t"] * (0.703487) + 32.807283
    df['mortgage'] = np.where(df['mortgage'] <= 10000, np.where(df['mortgage'] < 250, 250, df['mortgage']), 10000)
    df['HPE_Housing_Payment_Estimation'] = np.where(df["rent"] > df["mortgage"], df["rent"], df["mortgage"])
    return np.array(df['HPE_Housing_Payment_Estimation'])
    
    
#     output = df.withColumn("V71_G208S", when(df['V71_G208S'].isNull(),-5).otherwise(col("V71_G208S")))\
#                .withColumn("rent", col(median_rental_var) * exp(col(income_var) * (5.017e-06) + col("V71_AT36S") * (-0.0001028) + 
#                                                                 col("V71_G208S") * (-0.001778) + col("V71_BC03S") * (-0.01558) + 
#                                                                 col("W55_TRV14") * (0.007818) + col("V71_RT35S") * (3.578e-05) - 0.897))\
#                .withColumn("rent", when(col("rent") > 10000, 10000).when(col("rent") < 250, 250).otherwise(col("rent")))\
#                .withColumn("V71_MT01S_t", when(df['V71_MT01S'].isNull(), 0).otherwise(col("V71_MT01S")))\
#                .withColumn("W49_ATTR09_t", when(df['W49_ATTR09'].isNull(), 0).otherwise(col("W49_ATTR09")))\
#                .withColumn("mortgage", col(income_var) * (0.003352) + col("V71_MT01S_t") * (17.987879) + col("W49_ATTR09_t") * (0.703487) + 32.807283)\
#                .withColumn("mortgage", when(col("mortgage") > 10000, 10000).when(col("mortgage") < 250, 250).otherwise(col("mortgage")))\
#                .withColumn("HPE_Housing_Payment_Estimation", when(col("rent") > col("mortgage"), col("rent")).otherwise(col("mortgage")))\
#                .drop("V71_MT01S_t","W49_ATTR09_t","rent","mortgage")


    #return output.select('HPE_Housing_Payment_Estimation')

# COMMAND ----------

# # Construct and save the model
# model_path = folder_path + "Models/HPE_v2_pyfunc"
# hpe = HousingPaymentEstimator()
# mlflow.pyfunc.save_model(path=model_path, python_model=hpe)

# # Load the model in `python_function` format
# loaded_model = mlflow.pyfunc.load_model(model_path)

# Evaluate the model
# df = spark.read.csv('/mnt/science/Shringar/Databricks_Deployment/sample_126_scored.csv', header=True, inferSchema=True).withColumnRenamed("PIE_v5", "PIE_Income_Estimation")
hpev2 = loaded_model.predict(df2)

# COMMAND ----------

mlflow.pyfunc.log_model("HPE_v2_pyfunc_model3", python_model=HousingPaymentEstimator())

# COMMAND ----------

model_version_uri = f"models:/DM_HPE_v2/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

test['PIE_Income_Estimation'] = test['PIE_v5']
model_version_1.predict(test)

# COMMAND ----------

test['PHE_v2']

# COMMAND ----------

# MAGIC %md ### PIE_v5

# COMMAND ----------

model_path = folder_path + 'Models/pie_v5_xgb_final.json'
pie_features_path = folder_path + 'Features/piev5_features.csv'
artifacts = {"piev5_xgb_model": model_path, "features":pie_features_path}

# COMMAND ----------

class custom_piev5(mlflow.pyfunc.PythonModel):
  
  def __init__(self):
    super().__init__()

  def load_context(self, context):
    import xgboost as xgb
    self.xgb_model = xgb.Booster()
    self.xgb_model.load_model(context.artifacts["piev5_xgb_model"])
    #print(context.artifacts["piev5_xgb_model"])

  def predict(self, context, model_input):
    #print('This is newest version')
    input_matrix = xgb.DMatrix(model_input)
    return self.xgb_model.predict(input_matrix)

# COMMAND ----------

import xgboost as xgb
from sys import version_info

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

import cloudpickle
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'xgboost=={}'.format(xgb.__version__),
          'cloudpickle=={}'.format(cloudpickle.__version__),
        ],
      },
    ],
    'name': 'xgb_env'
}

# COMMAND ----------

#mlflow_pyfunc_model_path = folder_path + "Models/piev5_xgb_mlflow_pyfunc_model"
#mlflow.pyfunc.save_model(path=mlflow_pyfunc_model_path, python_model=custom_piev5(), artifacts=artifacts, conda_env=conda_env)
mlflow.pyfunc.log_model("pie_v5_pyfunc", python_model=custom_piev5(), artifacts=artifacts)

# COMMAND ----------

# MAGIC %md ### LAE_v3

# COMMAND ----------

df = pd.read_csv(folder_path + 'sample_126_w_ext_data_scored.csv')
fts = list(pd.read_csv(folder_path + 'Features/Version_Agnostic_Features/laev3_version_agnostic_features.csv', header=None)[0])
fts

# COMMAND ----------

lae = xgb.Booster()
lae.load_model(folder_path + "Models/Version_Agnostic_Models/lae_v3_final_version_agnostic.json")
all([a==b for a,b in zip(lae.feature_names,fts)])

# COMMAND ----------

model_path = folder_path + "Models/Version_Agnostic_Models/lae_v3_final_version_agnostic.json"
lae_features_path = folder_path + 'Features/Version_Agnostic_Features/laev3_version_agnostic_features.csv'
artifacts = {"laev3_xgb_model": model_path, "features":lae_features_path}

# COMMAND ----------

class custom_laev3(mlflow.pyfunc.PythonModel):
  
  def __init__(self):
    super().__init__()

  def load_context(self, context):
    import xgboost as xgb
    self.xgb_model = xgb.Booster()
    self.xgb_model.load_model(context.artifacts["laev3_xgb_model"])
    #print(context.artifacts["piev5_xgb_model"])

  def predict(self, context, model_input):
    #print('This is newest version')
    input_matrix = xgb.DMatrix(model_input)
    return self.xgb_model.predict(input_matrix)

# COMMAND ----------

mlflow.pyfunc.log_model("lae_v3_pyfunc", python_model=custom_laev3(), artifacts=artifacts)

# COMMAND ----------

# MAGIC %md ### CM_v6

# COMMAND ----------

import xgboost as xgb
cmv6 = xgb.Booster()
cmv6.load_model(folder_path + 'CM_v6_model/model.xgb')

# COMMAND ----------

cm_features_path = "/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/credit_model_v6_variables.csv"
cm_features = list(pd.read_csv(cm_features_path, header=None)[0])
cm_features = ['_'.join(col.split('_',2)[:2]) for col in cm_features]
cm_features

# COMMAND ----------

[col for col in cm_features]

# COMMAND ----------

model_path = folder_path + 'CM_v6_model/model.xgb'
#cmv6_features_path = folder_path + 'Features/credit_model_v6_variables.csv'
artifacts = {"cmv6_xgb_model": model_path}

# COMMAND ----------

class custom_cmv6(mlflow.pyfunc.PythonModel):
  
  def __init__(self, cm_features):
    self.cm_features = cm_features
    super().__init__()
    
  def load_context(self, context):
    import xgboost as xgb
    self.xgb_model = xgb.Booster()
    self.xgb_model.load_model(context.artifacts["cmv6_xgb_model"])
    #print(context.artifacts["piev5_xgb_model"])

  def predict(self, context, input_df):
    cm_features = self.cm_features
    #input_df = df.copy(deep=True)
    # Peprocessing
    impute_list = [-6, -5, -4, -3, -2, -1]
    cols_to_impute = ['W55_TRV07','W55_TRV08','W55_TRV09','W55_TRV10','W55_AGG909','W55_TRV17','V71_AT36S','V71_AU36S','V71_IN36S','V71_MT36S','V71_ST36S','W55_TRV01',
                      'W55_PAYMNT08','V71_AT21S','V71_BC21S','V71_BR21S']
    
    input_df[cols_to_impute] = np.vectorize(lambda x: 999 if x in impute_list else x)(input_df[cols_to_impute])
    
    input_df['DaysSinceMostRecentUILInquiry'] = np.where(input_df['epay01attr26'].isin(impute_list), 1e+05, input_df['epay01attr26'])
    input_df['MaxUtilizationOfUnsecuredInstallmentLoan'] = np.where(input_df['epay01attr21'].isin(impute_list), 0, input_df['epay01attr21'])
    input_df['UnsecuredInstallmentLoanBalance'] = np.where(input_df['epay01attr19'].isin(impute_list), 0, input_df['epay01attr19'])
    input_df['UnsecuredInstallmentLoanTrades'] = np.where(input_df['epay01attr22'].isin(impute_list), 0, input_df['epay01attr22'])
    input_df['UnsecuredInstallmentTradesOpenedLast12mo'] = np.where(input_df['epay01attr23'].isin(impute_list), 0, input_df['epay01attr23'])

    input_matrix = xgb.DMatrix(input_df[cm_features])
    
    return self.xgb_model.predict(input_matrix)

# COMMAND ----------

mlflow.pyfunc.log_model("cm_v6_pyfunc4", python_model=custom_cmv6(cm_features), artifacts=artifacts)

# COMMAND ----------

cm_vars = list(pd.read_csv(folder_path + 'Features/credit_model_v6_variables.csv', header=None)[0])
cm_vars = ['_'.join(col.split('_',2)[:2]) for col in cm_vars]
substitute = {'MaxUtilizationOfUnsecuredInstallmentLoan':'epay01attr21','DaysSinceMostRecentUILInquiry':'epay01attr26','UnsecuredInstallmentTradesOpenedLast12mo':'epay01attr23','UnsecuredInstallmentLoanTrades':'epay01attr22','UnsecuredInstallmentLoanBalance':'epay01attr19'}
# for idx, var in enumerate(cm_vars):
#   if var in substitute.keys():
#     cm_vars[idx] = substitute[var]
# cm_vars

# COMMAND ----------

temp_cols = [c for c in cm_features if c in df.columns]+[c for c in df.columns if 'epay' in c]

# COMMAND ----------

from pyspark.sql.functions import struct, col, when
df = spark.read.csv("/mnt/science/Shringar/Databricks_Deployment/sample_126_w_ext_data_scored.csv", header=True, inferSchema=True).select(*temp_cols)#.toPandas()
model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/DM_CM_v6/1", result_type='double')
df.withColumn("CM_Credit_Model_Score", model(struct(*map(col, df.columns)))).select('CM_Credit_Model_Score').display()

# COMMAND ----------

model_version_uri = f"models:/DM_CM_v6/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
cm = mlflow.pyfunc.load_model(model_version_uri)
df['CM_Credit_Model_Score'] = cm.predict(df)

# COMMAND ----------

df

# COMMAND ----------

class testClass:
  def __init__(self, name):
    self.name = name
    
  def testMethod(self, version):
    print(version)

class inheritClass(testClass):
  def testMethod(self, ver = '2'):
    print(ver)

# COMMAND ----------

cm_fts = list(pd.read_csv(folder_path + "Features/credit_model_v6_variables.csv", header=None)[0])

# COMMAND ----------

# model_name = "CM_v6"
# model_version = 1

# model = mlflow.xgboost.load_model(
#     model_uri=f"models:/{model_name}/{model_version}"
# )

mlflow.xgboost.save_model(model, "/dbfs/mnt/science/Shringar/Databricks_Deployment/cm_v6_mlflow")

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

model_uri = MlflowClient.get_model_version_download_uri("CM_v6", "1")
ModelsArtifactRepository(model_uri).download_artifacts(artifact_path="")

# COMMAND ----------

model_path = "/dbfs/mnt/science/Shringar/Databricks_Deployment/cm_v6_mlflow"
artifacts = {"cmv6_xgb_model": model_path}

# COMMAND ----------

class custom_cmv6(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import xgboost as xgb
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(context.artifacts["cmv6_xgb_model"])

    def predict(self, context, model_input):
        input_matrix = xgb.DMatrix(model_input.values)
        return self.xgb_model.predict(input_matrix)

# COMMAND ----------

import cloudpickle
from sys import version_info
import xgboost as xgb

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)

conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      'python={}'.format(PYTHON_VERSION),
      'pip',
      {
        'pip': [
          'mlflow',
          'xgboost=={}'.format(xgb.__version__),
          'cloudpickle=={}'.format(cloudpickle.__version__),
        ],
      },
    ],
    'name': 'xgb_env'
}

# COMMAND ----------

mlflow_pyfunc_model_path = '/dbfs/mnt/science/Shringar/Databricks_Deployment/cm_v6_mlflow_test'
mlflow.pyfunc.save_model(path=mlflow_pyfunc_model_path, python_model=custom_cmv6(), artifacts=artifacts, conda_env=conda_env)
mlflow.pyfunc.log_model("cmv6_xgb_mlflow_pyfunc2", python_model=custom_cmv6())

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

# COMMAND ----------

df_cm = df.copy(deep=True)
df_cm = spark.createDataFrame(df_cm)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, udf, when
def treat_attributes_credit_model_CPv6(df):
  output = (df
            .withColumn("W55_TRV07", F.when((df.W55_TRV07.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV07")))
            .withColumn("W55_TRV08", F.when((df.W55_TRV08.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV08")))
            .withColumn("W55_TRV09", F.when((df.W55_TRV09.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV09")))
            .withColumn("W55_TRV10", F.when((df.W55_TRV10.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV10")))
            
            .withColumn("W55_AGG909", F.when((df.W55_AGG909.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_AGG909")))
            .withColumn("W55_TRV17", F.when((df.W55_TRV17.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV17")))
            
            .withColumn("V71_AT36S", F.when((df.V71_AT36S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_AT36S")))
            .withColumn("V71_AU36S", F.when((df.V71_AU36S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_AU36S")))
            .withColumn("V71_IN36S", F.when((df.V71_IN36S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_IN36S")))
            .withColumn("V71_MT36S", F.when((df.V71_MT36S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_MT36S")))
            .withColumn("V71_ST36S", F.when((df.V71_ST36S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_ST36S")))
            .withColumn("W55_TRV01", F.when((df.W55_TRV01.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV01")))
            
            .withColumn("W55_PAYMNT08", F.when((df.W55_PAYMNT08.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_PAYMNT08")))
            .withColumn("V71_AT21S", F.when((df.V71_AT21S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_AT21S")))
            .withColumn("V71_BC21S", F.when((df.V71_BC21S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_BC21S")))
            .withColumn("V71_BR21S", F.when((df.V71_BR21S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_BR21S")))
            
            .withColumn("DaysSinceMostRecentUILInquiry",            F.when((df.epay01attr26.isin(-6, -5, -4, -3, -2, -1)), 1e+05 ).otherwise(F.col("epay01attr26")))
            .withColumn("MaxUtilizationOfUnsecuredInstallmentLoan", F.when((df.epay01attr21.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr21")))
            .withColumn("UnsecuredInstallmentLoanBalance",          F.when((df.epay01attr19.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr19")))
            .withColumn("UnsecuredInstallmentLoanTrades",           F.when((df.epay01attr22.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr22")))
            .withColumn("UnsecuredInstallmentTradesOpenedLast12mo", F.when((df.epay01attr23.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr23")))
            .withColumn("V71_AT104S_pct_all_trd_opned_pst_24mo_all_trd",                      F.col("V71_AT104S"))
            .withColumn("V71_AT31S_pct_opn_trd_gt_75pct_credln_verif_pst_12mo",               F.col("V71_AT31S"))
            .withColumn("V71_AT34B_util_for_opn_trd_verif_pst_12mo_excl_mtg_home_equity",     F.col("V71_AT34B"))
            .withColumn("V71_AT36S_mo_since_most_recent_dq",                                  F.col("V71_AT36S"))
            .withColumn("V71_BC09S_num_cc_trd_opned_pst_24mo",                                F.col("V71_BC09S"))
            .withColumn("V71_BC102S_avg_credln_opn_cc_trd_verif_pst_12mo",                    F.col("V71_BC102S"))
            .withColumn("V71_BC104S_avg_the_opn_cc_trd_utils_verif_pst_12mo",                 F.col("V71_BC104S"))
            .withColumn("V71_BC20S_mo_since_oldest_cc_trd_opned",                             F.col("V71_BC20S"))
            .withColumn("V71_BC21S_mo_since_most_recent_cc_trd_opned",                        F.col("V71_BC21S"))
            .withColumn("V71_BC31S_pct_opn_cc_trd_gt_75pct_credln_verif_pst_12mo",            F.col("V71_BC31S"))
            .withColumn("V71_BC97A_total_opn_buy_opn_cc_verif_pst_3mo",                       F.col("V71_BC97A"))
            .withColumn("V71_BC98A_total_opn_buy_opn_cc_verif_pst_12mo",                      F.col("V71_BC98A"))
            .withColumn("V71_BR20S_mo_since_oldest_bank_rvl_trd_opned",                       F.col("V71_BR20S"))
            .withColumn("V71_BR31S_pct_opn_bank_rvl_trd_gt_75pct_credln_verif_pst_12mo",      F.col("V71_BR31S"))
            .withColumn("V71_FI34S_util_for_opn_fin_instlmnt_trd_verif_pst_12mo",             F.col("V71_FI34S"))
            .withColumn("V71_G001B_num_30_or_more_dpd_ratings_pst_12mo",                      F.col("V71_G001B"))
            .withColumn("V71_G001S_num_30dpd_ratings_pst_12mo",                               F.col("V71_G001S"))
            .withColumn("V71_G201A_total_opn_buy_opn_trd_verif_pst_3mo_excl_instlmnt_mtg",    F.col("V71_G201A"))
            .withColumn("V71_G202A_total_opn_buy_opn_trd_verif_pst_12mo_excl_instlmnt_mtg",   F.col("V71_G202A"))
            .withColumn("V71_G242F_num_fin_inq_includes_dup_pst_3mo",                         F.col("V71_G242F"))
            .withColumn("V71_G242S_num_inq_includes_dup_pst_3mo",                             F.col("V71_G242S"))
            .withColumn("V71_G243F_num_fin_inq_includes_dup_pst_6mo",                         F.col("V71_G243F"))
            .withColumn("V71_G244F_num_fin_inq_includes_dup_pst_12mo",                        F.col("V71_G244F"))
            .withColumn("V71_G250B_num_30dpd_or_worse_itm_pst_12mo_excl_med_collect_itm",     F.col("V71_G250B"))
            .withColumn("V71_G250C_num_30dpd_or_worse_itm_pst_24mo_excl_med_collect_itm",     F.col("V71_G250C"))
            .withColumn("V71_G960S_num_dedup_inq",                                            F.col("V71_G960S"))
            .withColumn("V71_G980S_num_dedup_inq_pst_6mo",                                    F.col("V71_G980S"))
            .withColumn("V71_G990S_num_dedup_inq_pst_12mo",                                   F.col("V71_G990S"))
            .withColumn("V71_IN36S_mo_since_most_recent_instlmnt_dq",                         F.col("V71_IN36S"))
            .withColumn("V71_RE102S_avg_credln_opn_rvl_trd_verif_pst_12mo",                   F.col("V71_RE102S"))
            .withColumn("V71_RE31S_pct_opn_rvl_trd_gt_75pct_credln_verif_pst_12mo",           F.col("V71_RE31S"))
            .withColumn("V71_S004S_avg_num_mo_trd_have_been_on_file",                         F.col("V71_S004S")) 
            .withColumn("V71_S114S_num_dedup_inq_pst_6mo_excl_auto_mtg_inq",                  F.col("V71_S114S"))
            .withColumn("V71_S204S_total_bal_third_party_collect_verif_pst_12mo",             F.col("V71_S204S"))
            .withColumn("W55_AGG909_mo_since_max_agg_bnkcrd_bal_over_last_12mo",              F.col("W55_AGG909"))
            .withColumn("W55_AGG910_max_agg_bnkcrd_util_over_last_3mo",                       F.col("W55_AGG910"))
            .withColumn("W55_AGGS904_peak_mo_bnkcrd_spend_over_pst_12mo",                     F.col("W55_AGGS904"))
            .withColumn("W55_BALMAG01_non_mtg_bal_magnitude",                                 F.col("W55_BALMAG01"))
            .withColumn("W55_BALMAG02_rvl_bal_magnitude",                                     F.col("W55_BALMAG02"))
            .withColumn("W55_INDEX01_annual_yoy_spend_index",                                 F.col("W55_INDEX01"))
            .withColumn("W55_INDEX02_most_recent_quarter_yoy_spend_index",                    F.col("W55_INDEX02"))
            .withColumn("W55_PAYMNT08_ratio_actual_min_pmt_for_rvl_trd_last_mo",              F.col("W55_PAYMNT08"))
            .withColumn("W55_PAYMNT10_num_pmt_last_3mo",                                      F.col("W55_PAYMNT10"))
            .withColumn("W55_PAYMNT11_num_pmt_last_12mo",                                     F.col("W55_PAYMNT11"))
            .withColumn("W55_REVS904_max_agg_rvl_mo_spend_over_last_12mo",                    F.col("W55_REVS904"))
            .withColumn("W55_RVDEX01_annual_yoy_rvl_spend_index",                             F.col("W55_RVDEX01"))
            .withColumn("W55_RVDEX02_most_recent_quarter_yoy_rvl_spend_index",                F.col("W55_RVDEX02"))
            .withColumn("W55_RVLR01_util_for_bnkcrd_acct_with_a_rvl_bal",                     F.col("W55_RVLR01"))
            .withColumn("W55_TRV01_num_mo_since_overlimit_on_a_bnkcrd",                       F.col("W55_TRV01"))
            .withColumn("W55_TRV02_num_mo_overlimit_on_a_bnkcrd_over_last_12mo",              F.col("W55_TRV02"))
            .withColumn("W55_TRV03_num_non_mtg_trd_with_a_bal_incr_last_mo",                  F.col("W55_TRV03"))
            .withColumn("W55_TRV04_num_non_mtg_bal_incr_last_3mo",                            F.col("W55_TRV04"))
            .withColumn("W55_TRV07_num_non_mtg_bal_decr_last_mo",                             F.col("W55_TRV07"))
            .withColumn("W55_TRV08_num_non_mtg_bal_decr_last_3mo",                            F.col("W55_TRV08"))
            .withColumn("W55_TRV09_num_non_mtg_bal_decr_yoy",                                 F.col("W55_TRV09"))
            .withColumn("W55_TRV10_num_mo_non_mtg_bal_decr_last_12mo",                        F.col("W55_TRV10"))
            .withColumn("W55_TRV11_num_rvl_high_cred_incr_last_mo",                           F.col("W55_TRV11"))
            .withColumn("W55_TRV12_num_rvl_high_cred_incr_last_3mo",                          F.col("W55_TRV12"))
            .withColumn("W55_TRV13_num_rvl_high_cred_incr_yoy",                               F.col("W55_TRV13"))
            .withColumn("W55_TRV22_num_mo_bnkcrd_cred_limit_decr_last_12mo",                  F.col("W55_TRV22"))
            .withColumn("V71_AT06S_num_trd_opned_pst_6mo",                                    F.col("V71_AT06S"))
            .withColumn("V71_AT09S_num_trd_opned_pst_24mo",                                   F.col("V71_AT09S"))
            .withColumn("V71_AT21S_mo_since_most_recent_trd_opned",                           F.col("V71_AT21S"))
            .withColumn("V71_AT28A_total_credln_opn_trd_verif_pst_12mo",                      F.col("V71_AT28A"))
            .withColumn("V71_AT30S_pct_opn_trd_more_than_50pct_credln_verif_pst_12mo",        F.col("V71_AT30S"))
            .withColumn("V71_AT32S_max_bal_owed_on_opn_trd_verif_pst_12mo",                   F.col("V71_AT32S"))
            .withColumn("V71_AU36S_mo_since_most_recent_auto_dq",                             F.col("V71_AU36S"))       
            .withColumn("V71_BC34S_util_for_opn_cc_trd_verif_pst_12mo",                       F.col("V71_BC34S"))
            .withColumn("V71_BC35S_avg_bal_opn_cc_trd_verif_pst_12mo",                        F.col("V71_BC35S"))
            .withColumn("V71_BR09S_num_bank_rvl_trd_opned_pst_24mo",                          F.col("V71_BR09S"))
            .withColumn("V71_BR21S_mo_since_most_recent_bank_rvl_trd_opned",                  F.col("V71_BR21S"))
            .withColumn("V71_FI06S_num_fin_instlmnt_trd_opned_pst_6mo",                       F.col("V71_FI06S"))
            .withColumn("V71_FI09S_num_fin_instlmnt_trd_opned_pst_24mo",                      F.col("V71_FI09S"))
            .withColumn("V71_FI30S_pct_opn_fin_instlmnt_trd_gt_50pct_credln_verif_pst_12mo",  F.col("V71_FI30S"))
            .withColumn("V71_FI31S_pct_opn_fin_instlmnt_trd_gt_75pct_credln_verif_pst_12mo",  F.col("V71_FI31S"))
            .withColumn("V71_G058S_num_trd_30_or_more_dpd_pst_6mo",                           F.col("V71_G058S"))
            .withColumn("V71_G059S_num_trd_30_or_more_dpd_pst_12mo",                          F.col("V71_G059S"))
            .withColumn("V71_G061S_num_trd_30_or_more_dpd_pst_24mo",                          F.col("V71_G061S"))
            .withColumn("V71_G213A_highest_bal_third_party_collect_verif_24mo",               F.col("V71_G213A"))
            .withColumn("V71_G213B_highest_bal_non_med_third_party_collect_verif_24mo",       F.col("V71_G213B"))
            .withColumn("V71_G215A_num_third_party_collect_with_bal_larger_than_0_dollar",    F.col("V71_G215A"))
            .withColumn("V71_G234S_num_day_with_inquiry_occurring_pst_30day",                 F.col("V71_G234S"))
            .withColumn("V71_G237S_num_inq_pst_6mo_includes_dup",                             F.col("V71_G237S"))
            .withColumn("V71_G238S_num_inq_pst_12mo_includes_dup",                            F.col("V71_G238S"))
            .withColumn("V71_G244S_num_inq_pst_12mo_includes_dup",                            F.col("V71_G244S"))
            .withColumn("V71_G251A_num_60dpd_or_worse_itm_ever_excl_med_collect_itm",         F.col("V71_G251A"))
            .withColumn("V71_G251B_num_60dpd_or_worse_itm_pst_12mo_excl_med_collect_itm",     F.col("V71_G251B"))
            .withColumn("V71_G310S_worst_rating_on_all_trd_pst_12mo",                         F.col("V71_G310S"))
            .withColumn("V71_MT20S_mo_since_oldest_mtg_trd_opned",                            F.col("V71_MT20S"))
            .withColumn("V71_MT28S_total_credln_opn_mtg_trd_verif_pst_12mo",                  F.col("V71_MT28S"))
            .withColumn("V71_MT36S_mo_since_most_recent_mtg_dq",                              F.col("V71_MT36S"))
            .withColumn("V71_PB20S_mo_since_oldest_premium_cc_trd_opned",                     F.col("V71_PB20S"))
            .withColumn("V71_RE09S_num_rvl_trd_opned_pst_24mo",                               F.col("V71_RE09S"))
            .withColumn("V71_RE20S_mo_since_oldest_rvl_trd_opned",                            F.col("V71_RE20S"))
            .withColumn("V71_RE28S_total_credln_opn_rvl_trd_verif_pst_12mo",                  F.col("V71_RE28S"))
            .withColumn("V71_RE30S_pct_opn_rvl_trd_gt_50pct_credln_verif_pst_12mo",           F.col("V71_RE30S"))
            .withColumn("V71_RE34S_util_for_opn_rvl_trd_verif_pst_12mo",                      F.col("V71_RE34S"))
            .withColumn("V71_RT31S_pct_opn_retail_trd_gt_75pct_credln_verif_pst_12mo",        F.col("V71_RT31S"))
            .withColumn("V71_S043S_num_opn_trd_gt_50pct_credln_verif_pst_12mo_excl_instlmnt_mtg",F.col("V71_S043S"))
            .withColumn("V71_ST36S_mo_since_most_recent_student_ln_dq",                       F.col("V71_ST36S"))
            .withColumn("V71_ST99S_total_bal_all_student_ln_trd_ever_dq",                     F.col("V71_ST99S"))
            .withColumn("W55_AGG904_num_agg_non_mtg_cred_limit_decr_over_last_quarter",       F.col("W55_AGG904"))
            .withColumn("W55_AGG911_max_agg_bnkcrd_util_over_last_year",                      F.col("W55_AGG911"))
            .withColumn("W55_REVS901_agg_rvl_mo_spend_over_last_3mo",                         F.col("W55_REVS901"))
            .withColumn("W55_TRV17_num_bnkcrd_acct_with_a_yoy_cred_limit_incr",               F.col("W55_TRV17"))
            .withColumn("W55_TRV21_num_bnkcrd_acct_with_a_yoy_cred_limit_decr",               F.col("W55_TRV21"))
            .withColumn("V71_AT20S_mo_since_oldest_trd_opn",                                  F.col("V71_AT20S"))
            
    )
  return output

# COMMAND ----------

df_cm = treat_attributes_credit_model_CPv6(df_cm)

# COMMAND ----------

df_cm = df_cm.select(cm_fts)

# COMMAND ----------

mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

# COMMAND ----------


