# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
#from xgboost.sklearn import XGBRegressor
import sklearn.metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sklearn

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sn

folder_path = "/dbfs/mnt/science/Shringar/Databricks_Deployment/"

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

version_agnostic_models_path = folder_path + 'Models/Version_Agnostic_Models/'
[f for f in os.listdir(version_agnostic_models_path) if os.path.isfile(os.path.join(version_agnostic_models_path, f))]

# COMMAND ----------

# MAGIC %md ## Making models features version agnostic - input features and model score

# COMMAND ----------

# MAGIC %md ### PIE_v5

# COMMAND ----------

# Load old saved model
piev5 = xgb.Booster()
piev5.load_model(folder_path + 'Models/pie_v5_xgb_final.json')
piev5.feature_names

# COMMAND ----------

# Modifying feature names to remove year-specific information from external variables
piev5.feature_names = [col.replace('_2020', '') if col in ['acs_median_income_2020','acs_median_hh_income_2020'] else col for col in piev5.feature_names]
piev5.feature_names = [col.replace('_2010', '') if col == 'population_2010' else col for col in piev5.feature_names]
piev5.feature_names

# Re-saving model object
piev5.save_model(folder_path + 'Models/pie_v5_xgb_final_version_agnostic.json')

# COMMAND ----------

# Evaluating newly saved model and feature names
piev5 = xgb.Booster()
piev5.load_model(folder_path + 'Models/pie_v5_xgb_final_version_agnostic.json')
piev5.feature_names

# COMMAND ----------

# Saving feature names to csv file
piev5_fts = pd.Series(piev5.feature_names, name='piev5_fts')
piev5_fts.to_csv(folder_path + 'Features/Version_Agnostic_Features/piev5_version_agnostic_features.csv', index=False, header=False)

# COMMAND ----------

# MAGIC %md ### LAE_v3

# COMMAND ----------

## Load model
laev3 = xgb.Booster()
laev3.load_model(folder_path + 'Models/bst_LA_v3_resaving.model')

# Loading LAE V3 features from csv file
laev3_vars = list(pd.read_csv(folder_path + 'Features/LAE_v3_variables.csv', header=None)[0])

# Renaming features
laev3_vars = ['PIE_Income_Estimation' if col == 'PIE_v4' else col for col in laev3_vars]

laev3.feature_names = laev3_vars
print(laev3.feature_names)

# Re-saving model object
laev3.save_model(folder_path + 'Models/lae_v3_final_version_agnostic.json')

# COMMAND ----------

# Evaluating newly saved model and feature names
laev3 = xgb.Booster()
laev3.load_model(folder_path + 'Models/Version_Agnostic_Models/lae_v3_final_version_agnostic.json')
laev3.feature_names

# COMMAND ----------

# Saving feature names to csv file
laev3_fts = pd.Series(laev3.feature_names, name='laev3_fts')
laev3_fts.to_csv(folder_path + 'Features/Version_Agnostic_Features/laev3_version_agnostic_features.csv', index=False, header=False)

# COMMAND ----------

laev3.get_score(importance_type='gain')

# COMMAND ----------

# MAGIC %md ### CM6

# COMMAND ----------

cm6 = xgb.Booster()
cm6.load_model(folder_path + 'CM_v6_model/model.xgb')

# COMMAND ----------

cm6_vars = list(pd.read_csv(folder_path + 'Features/credit_model_v6_variables.csv', header=None)[0])
cm6_vars = ['_'.join(col.rsplit('_')[:2]) for col in cm6_vars]
cm6_vars


# COMMAND ----------

cm6 = mlflow.pyfunc.spark_udf(spark, model_uri="models:/CM_v6/1")

# COMMAND ----------

cm6

# COMMAND ----------

import mlflow.pyfunc
#mlflow.pyfunc.load_model(folder_path + 'Models/bst_LA_v3_request.model')
cm6 = xgb.Booster()
cm6.load_model(folder_path + 'Models/credit_model_v6.json')

# COMMAND ----------

# MAGIC %md ### RR_v9

# COMMAND ----------

# MAGIC %r
# MAGIC # Reading RRv9 features from RDS file and writing to csv file
# MAGIC rrv9_vars <- readRDS("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/bst_RR_v9_36_vars.rds")
# MAGIC readr::write_csv(as.data.frame(rrv9_vars), "/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/RR_v9_features.csv")

# COMMAND ----------

# Load old saved model
rrv9 = xgb.Booster()
rrv9.load_model(folder_path + 'Models/bst_RR_v9_36.json')

# Loading RR V9 features from csv file
rrv9_vars = list(pd.read_csv(folder_path + 'Features/RR_v9_features.csv')['rrv9_vars'])
rrv9_vars

# Renaming features
rrv9_vars = ['CM_Credit_Model_Score' if col == 'Model_v6_Score' else col for col in rrv9_vars]

rrv9.feature_names = rrv9_vars
print(rrv9.feature_names)

# Re-saving model object
rrv9.save_model(folder_path + 'Models/rr_v9_final_version_agnostic.json')

# COMMAND ----------

# Saving feature names to csv file
rrv9_fts = pd.Series(rrv9.feature_names, name='rrv9_fts')
rrv9_fts.to_csv(folder_path + 'Features/Version_Agnostic_Features/rrv9_version_agnostic_features.csv', index=False, header=False)

# COMMAND ----------

# MAGIC %md ### RR_v10

# COMMAND ----------

# MAGIC %r
# MAGIC # Reading RRv10 features from RDS file and writing to csv file
# MAGIC rrv10_vars <- readRDS("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/bst_RR_v10_36_vars.rds")
# MAGIC readr::write_csv(as.data.frame(rrv10_vars), "/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/RR_v10_features.csv")

# COMMAND ----------

# Load old saved model
rrv10 = xgb.Booster()
rrv10.load_model(folder_path + 'Models/bst_RR_v10_36.json')

# Loading RR V10 features from csv file
rrv10_vars = list(pd.read_csv(folder_path + 'Features/RR_v10_features.csv')['rrv10_vars'])
rrv10_vars

# Renaming features
rrv10_vars = ['CM_Credit_Model_Score' if col == 'Model_v6_Score' else col for col in rrv10_vars]

rrv10.feature_names = rrv10_vars
print(rrv10.feature_names)

# Re-saving model object
rrv10.save_model(folder_path + 'Models/rr_v10_final_version_agnostic.json')

# COMMAND ----------

# MAGIC %md ### AR_v7

# COMMAND ----------

# Load old saved model
arv7 = xgb.Booster()
arv7.load_model(folder_path + 'Models/final_model_AR_v7.json')
arv7.feature_names

# COMMAND ----------

# Modifying feature names to remove year-specific information from external variables
arv7.feature_names = ['CM_Credit_Model_Score' if col == 'Model_v6_Score' else col for col in arv7.feature_names]
arv7.feature_names = ['HPE_Housing_Payment_Estimation' if col == 'PHE_v2' else col for col in arv7.feature_names]
print(arv7.feature_names)

# Re-saving model object
arv7.save_model(folder_path + 'Models/ar_v7_final_version_agnostic.json')

# COMMAND ----------

# Evaluating newly saved model and feature names
arv7 = xgb.Booster()
arv7.load_model(folder_path + 'Models/ar_v7_final_version_agnostic.json')
print(arv7.feature_names)

# Saving feature names to csv file
arv7_fts = pd.Series(arv7.feature_names, name='arv7_fts')
arv7_fts.to_csv(folder_path + 'Features/Version_Agnostic_Features/arv7_version_agnostic_features.csv', index=False, header=False)

# COMMAND ----------

# MAGIC %md ### FR_v7

# COMMAND ----------

# Load old saved model
frv7 = xgb.Booster()
frv7.load_model(folder_path + 'Models/final_model_FRv7.json')
frv7.feature_names

# COMMAND ----------

# Modifying feature names to remove year-specific information from external variables
frv7.feature_names = ['CM_Credit_Model_Score' if col == 'Model_v6_Score' else col for col in frv7.feature_names]
frv7.feature_names = ['state_label_5_FR' if col == 'state_label_5_FR_v6' else col for col in frv7.feature_names]
print(frv7.feature_names)

# Re-saving model object
frv7.save_model(folder_path + 'Models/fr_v7_final_version_agnostic.json')

# COMMAND ----------

# Evaluating newly saved model and feature names
frv7 = xgb.Booster()
frv7.load_model(folder_path + 'Models/fr_v7_final_version_agnostic.json')
print(frv7.feature_names)

# Saving feature names to csv file
frv7_fts = pd.Series(frv7.feature_names, name='frv7_fts')
frv7_fts.to_csv(folder_path + 'Features/Version_Agnostic_Features/frv7_version_agnostic_features.csv', index=False, header=False)

# COMMAND ----------

# MAGIC %md ### Pre UOL

# COMMAND ----------

# Load old saved model
preuol = xgb.Booster()
preuol.load_model(folder_path + 'Models/pre_UOL_model_v1.json')

# Loading RR V9 features from csv file
preuol_vars = list(pd.read_csv(folder_path + 'Features/pre_UOL_model_v1_variables.csv', header=None)[0])

# Renaming features
preuol_vars = ['PIE_Income_Estimation' if col == 'PIE_v3' else col for col in preuol_vars]
preuol_vars

preuol.feature_names = preuol_vars
print(preuol.feature_names)

# Re-saving model object
preuol.save_model(folder_path + 'Models/Version_Agnostic_Models/preuol_v1_final_version_agnostic.json')

# COMMAND ----------

# Evaluating newly saved model and feature names
preuol = xgb.Booster()
preuol.load_model(folder_path + 'Models/Version_Agnostic_Models/preuol_v1_final_version_agnostic.json')
print(preuol.feature_names)

# Saving feature names to csv file
preuol_fts = pd.Series(preuol.feature_names, name='preuol_fts')
preuol_fts.to_csv(folder_path + 'Features/Version_Agnostic_Features/preuolv1_version_agnostic_features.csv', index=False, header=False)

# COMMAND ----------

# MAGIC %md ## Log all DM models on MLflow

# COMMAND ----------

# MAGIC %md ### PIE_v5

# COMMAND ----------

piev5_mlflow = xgb.Booster()
piev5_mlflow.load_model(version_agnostic_models_path + 'pie_v5_xgb_final_version_agnostic.json')
piev5_mlflow.feature_names

# COMMAND ----------

import mlflow
mlflow.xgboost.log_model(piev5_mlflow, "piev5_xgboost_model_version_agnostic")

# COMMAND ----------

mlflow.search_runs()['run_id']

# COMMAND ----------

import mlflow
mlflow.xgboost.log_model(piev5_mlflow, "piev5_xgboost_model_version_agnostic")

run_id = mlflow.search_runs()['run_id'][0]
piev5_uri = f"runs:/{run_id}/piev5_xgboost_model_version_agnostic"
model_details = mlflow.register_model(model_uri=piev5_uri, name="DM_PIE_v5_prod")

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name="DM_PIE_v5_prod", version=1)

print(model_version_details.status)

client.update_registered_model(name=model_details.name, description="Personal Income Estimator V5 used for income prediction at prescreen stage. The model is used as an input to multiple downstream prescreen models. Built using xgboost 1.5.2")

import time
time.sleep(10)

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

test = pd.read_csv("/dbfs/mnt/science/Chong/databricks/sample_data/sample_3_after_all_models.csv")
test

# COMMAND ----------

# MAGIC %md ### LAE_v3

# COMMAND ----------

laev3_mlflow = xgb.Booster()
laev3_mlflow.load_model(version_agnostic_models_path + 'lae_v3_final_version_agnostic.json')
laev3_mlflow.feature_names

# COMMAND ----------

mlflow.xgboost.log_model(laev3_mlflow, "laev3_xgboost_model_version_agnostic")

# COMMAND ----------

mlflow.search_runs()['run_id'][0]

# COMMAND ----------

run_id = mlflow.search_runs()['run_id'][0]
laev3_uri = f"runs:/{run_id}/laev3_xgboost_model_version_agnostic"
model_details = mlflow.register_model(model_uri=laev3_uri, name="DM_LAE_v3_prod")

# COMMAND ----------

model_version_details

# COMMAND ----------

import mlflow
# mlflow.xgboost.log_model(laev3_mlflow, "piev5_xgboost_model_version_agnostic")

# run_id = mlflow.search_runs()['run_id'][0]
# laev3_uri = f"runs:/{run_id}/laev3_xgboost_model_version_agnostic"
# model_details = mlflow.register_model(model_uri=laev3_uri, name="DM_LAE_v3_prod")

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name="DM_LAE_v3_prod", version=1)

print(model_version_details.status)

client.update_registered_model(name="DM_LAE_v3_prod", description="Loan Amount Estimator V3 used for estimating the loan amount at prescreen stage. The model is used for multiple calculations and as an input to downstream prescreen models. Resaved in xgboost 1.5.2")

import time
time.sleep(10)

client.transition_model_version_stage(
    name="DM_LAE_v3_prod",
    version='1',
    stage="Production"
)

model_version_details = client.get_model_version(
  name="DM_LAE_v3_prod",
  version='1',
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

laev3_vars = list(pd.read_csv(folder_path + 'Features/LAE_v3_variables.csv', header=None)[0])
laev3_vars

# COMMAND ----------

set(laev3_vars).difference(set(test.columns))

# COMMAND ----------

list(test.columns)

# COMMAND ----------

[col for col in list(test.columns) if 'Loan' in col]

# COMMAND ----------

# MAGIC %md ### RR_v9

# COMMAND ----------

rrv9_mlflow = xgb.Booster()
rrv9_mlflow.load_model(version_agnostic_models_path + 'rr_v9_final_version_agnostic.json')
rrv9_mlflow.feature_names

# COMMAND ----------

import mlflow
#mlflow.xgboost.log_model(rrv9_mlflow, "rrv9_xgboost_model_version_agnostic")

run_id = mlflow.search_runs()['run_id'][0]
rrv9_uri = f"runs:/{run_id}/rrv9_xgboost_model_version_agnostic"
model_details = mlflow.register_model(model_uri=rrv9_uri, name="DM_RR_v9_prod")

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name="DM_RR_v9_prod", version=1)

print(model_version_details.status)

client.update_registered_model(name=model_details.name, description="Response Rate Model V9 used for predicting response at prescreen stage. Resaved in xgboost 1.5.2")

import time
time.sleep(10)

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

# MAGIC %md ### RR_v10

# COMMAND ----------

rrv10_mlflow = xgb.Booster()
rrv10_mlflow.load_model(version_agnostic_models_path + 'rr_v10_final_version_agnostic.json')
print(rrv10_mlflow.feature_names)

import mlflow
mlflow.xgboost.log_model(rrv10_mlflow, "rrv10_xgboost_model_version_agnostic")

# COMMAND ----------

pd.Series(rrv10_mlflow.feature_names).to_csv(folder_path + 'Features/Version_Agnostic_Features/rrv10_version_agnostic_features.csv', index=False, header=False)

# COMMAND ----------

pd.read_csv(folder_path + 'Features/Version_Agnostic_Features/rrv10_version_agnostic_features.csv', header=None)[0]

# COMMAND ----------

# MAGIC %md ### AR_v7

# COMMAND ----------

arv7_mlflow = xgb.Booster()
arv7_mlflow.load_model(version_agnostic_models_path + 'ar_v7_final_version_agnostic.json')
arv7_mlflow.feature_names

# COMMAND ----------

import mlflow
#mlflow.xgboost.log_model(arv7_mlflow, "arv7_xgboost_model_version_agnostic")

run_id = mlflow.search_runs()['run_id'][0]
arv7_uri = f"runs:/{run_id}/arv7_xgboost_model_version_agnostic"
model_details = mlflow.register_model(model_uri=arv7_uri, name="DM_AR_v7_prod")

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name="DM_AR_v7_prod", version=1)

print(model_version_details.status)

client.update_registered_model(name=model_details.name, description="Approval Rate Model V7 used for predicting at prescreen stage whether individual will be approved for loan application. Resaved in xgboost 1.5.2")

import time
time.sleep(10)

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

mlflow.search_runs()['run_id'][0]

# COMMAND ----------

# MAGIC %md ### FRv7

# COMMAND ----------

frv7_mlflow = xgb.Booster()
frv7_mlflow.load_model(version_agnostic_models_path + 'fr_v7_final_version_agnostic.json')
frv7_mlflow.feature_names

# COMMAND ----------

import mlflow
#mlflow.xgboost.log_model(frv7_mlflow, "frv7_xgboost_model_version_agnostic")

run_id = mlflow.search_runs()['run_id'][0]
frv7_uri = f"runs:/{run_id}/frv7_xgboost_model_version_agnostic"
model_details = mlflow.register_model(model_uri=frv7_uri, name="DM_FR_v7_prod")

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name="DM_FR_v7_prod", version=1)

print(model_version_details.status)

client.update_registered_model(name=model_details.name, description="Funding Rate Model V7 used for predicting at prescreen stage whether individual will get funded for loan or not. Built using xgboost 1.5.2")

import time
time.sleep(10)

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

mlflow.search_runs()['run_id']

# COMMAND ----------

# MAGIC %md ### Pre-UOL

# COMMAND ----------

preuol_mlflow = xgb.Booster()
preuol_mlflow.load_model(version_agnostic_models_path + 'preuol_v1_final_version_agnostic.json')
preuol_mlflow.feature_names

# COMMAND ----------

import mlflow
# mlflow.xgboost.log_model(preuol_mlflow, "preuol_xgboost_model_version_agnostic")

run_id = mlflow.search_runs()['run_id'][0]
preuol_uri = f"runs:/{run_id}/preuol_xgboost_model_version_agnostic"
model_details = mlflow.register_model(model_uri=preuol_uri, name="DM_preUOL_v1_prod")

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name="DM_preUOL_v1_prod", version=1)

print(model_version_details.status)

client.update_registered_model(name=model_details.name, description="Pre-UOL model used at prescreen stage. Resaved in xgboost 1.5.2")

import time
time.sleep(10)

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print(f"The current model stage is: '{model_version_details.current_stage}'")

# COMMAND ----------

mlflow.search_runs()['run_id']

# COMMAND ----------

# MAGIC %md ## Testing integrity of models in production on MLflow

# COMMAND ----------

test = pd.read_csv(folder_path + 'sample_126_scored.csv', dtype={'NCOA_ADDR_ZipCode':'string'}).rename(columns={'NCOA_ADDR_ZipCode':'zipcode'})
external_data = pd.read_csv(folder_path + 'piev5_external_data_modeling.csv', dtype={'zipcode':'string'})
test['zipcode'] = test['zipcode'].str.zfill(5)
external_data['zipcode'] = external_data['zipcode'].str.zfill(5)
test.head()

# COMMAND ----------

test = test.merge(external_data, how='left', on='zipcode')

# COMMAND ----------

test = test.rename(columns={'acs_median_income_2020':'acs_median_income', 'acs_median_hh_income_2020':'acs_median_hh_income', 'population_2010':'population'})

# COMMAND ----------

features_path = folder_path + 'Features/'
ver_ag_features_path = features_path + 'Version_Agnostic_Features/'

# COMMAND ----------

[col for col in test.columns if 'PIE' in col]

# COMMAND ----------

# MAGIC %md ### PIE_v5

# COMMAND ----------

piev5_fts = list(pd.read_csv(features_path + 'piev5_features.csv', index_col = 0)['piev5_fts'])
piev5_fts

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/DM_PIE_v5_prod/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

test['piev5_pred'] = pd.Series(model_version_1.predict(test[piev5_fts]))

# COMMAND ----------

all(np.where(test['PIE_v5'] == test['piev5_pred'], True, False))

# COMMAND ----------

# MAGIC %md ### LAE_v3

# COMMAND ----------

laev3_fts = list(pd.read_csv(features_path + 'LAE_v3_variables.csv', header=None)[0])
laev3_fts = ['PIE_v5' if col == 'PIE_v4' else col for col in laev3_fts]

# COMMAND ----------

laev3_fts_ver_ag = list(pd.read_csv(ver_ag_features_path + 'laev3_version_agnostic_features.csv', header=None)[0])
laev3_fts_ver_ag

# COMMAND ----------

test = test.rename(columns=dict(zip(laev3_fts, laev3_fts_ver_ag)))

# COMMAND ----------

[col for col in test.columns if 'PIE' in col]

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/DM_LAE_v3_prod/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

test['laev3_pred'] = pd.Series(model_version_1.predict(test[laev3_fts_ver_ag]))

all(np.where(test['PredLoanAmount_v3'].round(2) == test['laev3_pred'].round(2), True, False))

# COMMAND ----------

all([a==b for a,b in zip(test['laev3_pred'].round(), test['PredLoanAmount_v3'].round())])

# COMMAND ----------

test[['PredLoanAmount_v3','laev3_pred']][~np.isclose(test['PredLoanAmount_v3'], test['laev3_pred'], atol=1e-12, rtol=0)]

# COMMAND ----------

np.isclose(test['PredLoanAmount_v3'], test['laev3_pred'], rtol=0)

# COMMAND ----------

test[['PredLoanAmount_v3','laev3_pred']][np.where(test['PredLoanAmount_v3'] != test['laev3_pred'], True, False)]

# COMMAND ----------

# MAGIC %md ### RR_v9

# COMMAND ----------

import xgboost as xgb
fr = xgb.Booster()
fr.load_model("/dbfs/mnt/science/Shringar/Databricks_Deployment/Models/Version_Agnostic_Models/fr_v7_final_version_agnostic.json")
fr.feature_names

# COMMAND ----------

import mlflow
mlflow.xgboost.log_model(ar, "arv7_mlflow")

# COMMAND ----------

rrv9_fts = list(pd.read_csv(features_path + 'RR_v9_features.csv', header=0)['rrv9_vars'])
#rrv9_fts = ['PIE_v5' if col == 'PIE_v4' else col for col in laev3_fts]
rrv9_fts

# COMMAND ----------

rrv9_fts_ver_ag = list(pd.read_csv(ver_ag_features_path + 'rrv9_version_agnostic_features.csv', header=None)[0])
rrv9_fts_ver_ag

# COMMAND ----------

test = test.rename(columns=dict(zip(rrv9_fts, rrv9_fts_ver_ag)))

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/DM_RR_v9_prod/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

test['rrv9_pred'] = pd.Series(model_version_1.predict(test[rrv9_fts_ver_ag]))

all(np.where(test['ResponseProb_v9'].round(2) == test['rrv9_pred'].round(2), True, False))

# COMMAND ----------

test[['ResponseProb_v9','rrv9_pred']][~np.isclose(test['ResponseProb_v9'], test['rrv9_pred'], atol=1e-12, rtol=0)]

# COMMAND ----------

# MAGIC %md ### CM6

# COMMAND ----------

cm6_fts = list(pd.read_csv(features_path + 'credit_model_v6_variables.csv', header=None)[0])
cm6_fts

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### AR_v7

# COMMAND ----------

test = test.rename(columns={'PHE_v2':'HPE_Housing_Payment_Estimation', 'N05_S0Y2':'N05_S0_Y2'})

# COMMAND ----------

arv7_fts_ver_ag = list(pd.read_csv(ver_ag_features_path + 'arv7_version_agnostic_features.csv', header=None)[0])
#arv7_fts_ver_ag = ['N05_S0Y2' if col == 'N05_S0_Y2' else col for col in arv7_fts_ver_ag]
#arv7_fts_ver_ag = ['epay01attr24' if col == 'W49_ATTR24' else col for col in arv7_fts_ver_ag]
arv7_fts_ver_ag

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/DM_AR_v7_prod/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

test['arv7_pred'] = pd.Series(model_version_1.predict(test[arv7_fts_ver_ag]))

all(np.where(test['ApprovalProb_v6'] == test['arv7_pred'], True, False))

# COMMAND ----------

pd.read_csv()

# COMMAND ----------

# MAGIC %md ### PreUOL

# COMMAND ----------


