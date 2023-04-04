# Databricks notebook source
import numpy as np
import datetime as dt
import joblib

import pandas as pd
import re
from pyspark.sql.functions import col, lit, when
import pyspark.sql.functions as F

pd.set_option('display.max_rows', 500)

#folder_path = "/mnt/science/Shringar/Databricks_Deployment/"

# COMMAND ----------

# MAGIC %run ./ML_Deployment_Pipeline_Part_1

# COMMAND ----------

# MAGIC %run ./ML_Deployment_Pipeline_Part_2

# COMMAND ----------

# MAGIC %run ./ML_Deployment_Pipeline_Part_3

# COMMAND ----------

# MAGIC %run ./ML_Deployment_Pipeline_Part_4

# COMMAND ----------

final_fts = ['NCOA_ADDR_State','NCOA_ADDR_ZipCode','GGENERC_gna001','CEMP08_b_finscr','W55_AGG911','V71_AT03S','V71_AT09S', 'V71_AT101S','V71_AT20S','V71_G069S','V71_G095S','V71_MT01S','V71_MT101S',
                  'V71_S114S','V71_ST02S','N05_G082','V71_US02S','V71_US101S','RECENT_TAG','PREVIOUS_COUNTER','PPPCL_CONSUMER_ID','V71_ST101S','V71_AU02S','V71_IN02S','V71_G960S', 'V71_IN09S',
                  
                  'MeanIncome','MedianRental','mean_net_inv_income_tax','mean_state_sales_tax',
                  
                  'FederalTax','StateTax','MonthlyIncome','NDI',
                  
                  #'PredLoanAmount_gamboost','PredLoanAmount_gamboost2', 'PredLoanAmount_gamboost3',
                  'LAE_Loan_Amount_Estimation',
                  
                  'UILBalance', 'DaysSinceOpenedUIL', 'MaxUILUtilization','OpenUILAccounts', 'UILAccountsOpenedPast12mo','NonMortgageMonthlyPayment','RevolvingBalance','DaysSinceUILInquiry','Tradeline_NDI','UnsecuredSummaryBalance',
                  'LieDetectorRatio','LieDetectorDiff','LieDetectorHigh','NdiWithLieDetector','MonthlyDti','Bti','BtiWithLieDetector','PctNewBCTradesInPast12mo','PercentDelinquent','OverlimTrend','Mdti',
                  
                  #'Model_v5_Score','PricingTier_v5','Knockouts_v5','Decision_v5', 'Model_v5_5_Score','PricingTier_v5_5', 'Knockouts_v5_5','Decision_v5_5', 'CPv5_5_KO_W55_AGG911','CPv5_5_KO_V71_AT03S', 'CPv5_5_KO_V71_AT20S', 'CPv5_5_KO_V71_G069S', 
                  'CPv5_5_KO_V71_G095S', 'CPv5_5_KO_N05_G082', 'CPv5_5_KO_PctNewBCTradesInPast12mo', 
                  
                  #'Model_v3_Score', 
                  'CM_Credit_Model_Score', 'PricingTier_v6', #'PricingTier',
                  
                  #'ResponseProb_v3', 'ResponseProb_v4', 'ResponseProb_v4_corrected', 'ResponseProb_v5.1', 'ResponseProb_v5.1_cap', 'ResponseProb_v6', 'ResponseProb_v7', 
                  #'ResponseProb_v8', 'ResponseProb_v8_dm', 'ResponseProb_v9', 'ResponseProb', 'ResponseProb2', 'So_ResponseProb_v1',
                  'RR_ResponseProb',
                  
                  #'ApprovalProb_v4','ApprovalProb_v5','ApprovalProb_v6',
                  'AR_ApprovalProb',
                  
                  #'FundingProb_v41','FundingProb_v5', 'FundingProb_v6', #FundingProb2,
                  'FR_FundingProb'
                  
                  #'OriginationFee','all_premium_more_than_15k','all_premium_less_than_15k','all_prem_factor','all_prem','all_revenue','all_premium_more_than_15k_new', 'all_premium_less_than_15k_new',
                   #'all_prem_factor2' = all_prem_factor_nw_nl,all_prem2= all_prem_nw_nl,all_revenue2,
                  
                  'N05_S0Y2','HC03_VC113','no_student_trades',
                  
                  'eligible_last_month','email_only','last_applied_date','last_partner_applied_date','last_mailed_campaign_month','last_mailed_selection_month','months_since_last_application','lastreportedincome','lastreportedhousingpayment', 
                  'last_mailed_campaign_month', 'last_mailed_selection_month',
                  
                  #'pre_UOL_score_v1',
                  'preUOLScore',
                  
                  #'creative_v2_confetti_201910', 'creative_v2_black_back_202007',
                  
                  'ehpm01_nopdfor', 'ehpm01_nopmod', 'ehpm01_nopnd', 'ehpm01_nocpgt45', 'ehpm01_nocpay0', 'ehpm01_nocpay1', 'ehpm01_nocpgt45', 'ehpm01_nocpgt46',
                  
                  'V71_RE101S', 'V71_IN101S', 'V71_HR101S', 'V71_AU101S', 'epay01attr25', 'epay01attr19', 'eusp02score', 'sendableemail']

# COMMAND ----------

ar_fts = list(pd.read_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/ARv7_features.csv")['0'])
ar_fts = ['N05_S0Y2' if col == 'N05_S0_Y2' else col for col in ar_fts]
ar_fts = ['epay01attr24' if col == 'W49_ATTR24' else col for col in ar_fts]
ar_fts_new = list(pd.read_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/Version_Agnostic_Features/arv7_version_agnostic_features.csv", header=None)[0])

sample = spark.read.csv("/mnt/science/Shringar/Databricks_Deployment/sample_126_w_ext_data_scored.csv", header=True, inferSchema=True)
ar_fr_score = spark.read.csv("/mnt/dm_processed/scored/PSCamp126/dm_PS126_scored_AR_FR_score.csv", header=True, inferSchema=True)
sample = sample.join(ar_fr_score, how='left', on='PPPCL_CONSUMER_ID')
sample = sample.select(*(['PPPCL_CONSUMER_ID','ApprovalProb_v7'] + ar_fts)).toPandas()

df = spark.read.csv("/mnt/science/Shringar/Databricks_Deployment/scored_files/sample_126_scored.csv", header=True, inferSchema=True).select(*(['PPPCL_CONSUMER_ID','AR_ApprovalProb'] + ar_fts_new)).toPandas()

merged = df.merge(sample, on='PPPCL_CONSUMER_ID', how='left', suffixes=('_DB', '_R')).fillna(0)

# COMMAND ----------

merged.fillna(0)

# COMMAND ----------

combined_fts = [(a+'_DB', b+'_R') if a == b else (a, b) for a, b in list(zip(['AR_ApprovalProb']+ar_fts_new, ['ApprovalProb_v7']+ar_fts))]
diff = pd.DataFrame()
for a, b in combined_fts:
  diff[b.replace('_R','_diff')] = merged[a] - merged[b]
diff = diff.abs()

# COMMAND ----------

diff.corr()#[['ApprovalProb_v7']].abs().to_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/corr_abs_diff_ARv7.csv")

# COMMAND ----------

# MAGIC %md ### Differences in FR

# COMMAND ----------

fr_fts = list(pd.read_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/FRv7_features.csv")['0'])
#fr_fts = ['N05_S0Y2' if col == 'N05_S0_Y2' else col for col in ar_fts]
#fr_fts = ['epay01attr24' if col == 'W49_ATTR24' else col for col in ar_fts]
fr_fts_new = list(pd.read_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/Version_Agnostic_Features/frv7_version_agnostic_features.csv", header=None)[0])

sample = spark.read.csv("/mnt/science/Shringar/Databricks_Deployment/sample_126_w_ext_data_scored.csv", header=True, inferSchema=True)
ar_fr_score = spark.read.csv("/mnt/dm_processed/scored/PSCamp126/dm_PS126_scored_AR_FR_score.csv", header=True, inferSchema=True)
sample = sample.join(ar_fr_score, how='left', on='PPPCL_CONSUMER_ID')
sample = sample.select(*(['PPPCL_CONSUMER_ID','FundingProb_v7'] + fr_fts)).toPandas()

df = spark.read.csv("/mnt/science/Shringar/Databricks_Deployment/scored_files/sample_126_scored.csv", header=True, inferSchema=True).select(*(['PPPCL_CONSUMER_ID','FR_FundingProb'] + fr_fts_new)).toPandas()

merged = df.merge(sample, on='PPPCL_CONSUMER_ID', how='left', suffixes=('_DB', '_R')).fillna(0)

merged = merged[~merged['PPPCL_CONSUMER_ID'].isin(drop_rows)]

# COMMAND ----------

combined_fts = [(a+'_DB', b+'_R') if a == b else (a, b) for a, b in list(zip(['FR_FundingProb']+fr_fts_new, ['FundingProb_v7']+fr_fts))]
diff = pd.DataFrame()
for a, b in combined_fts:
  diff[b.replace('_R','_diff')] = merged[a] - merged[b]
diff = diff.abs()

# COMMAND ----------

diff['FundingProb_v7'].describe()

# COMMAND ----------

diff.corr()#[['FundingProb_v7']].abs().to_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/corr_abs_diff_FRv7.csv")

# COMMAND ----------

#merged[['W55_TRV08_DB','W55_TRV08_R']]
drop_rows = merged[['PPPCL_CONSUMER_ID','W55_TRV08_DB','W55_TRV08_R']][~np.isclose(merged['W55_TRV08_DB'], merged['W55_TRV08_R'], rtol=0)]['PPPCL_CONSUMER_ID'].unique()
drop_rows

# COMMAND ----------

diff[diff['PPPCL_CONSUMER_ID'].isin(drop_rows)]

# COMMAND ----------

# Reading PSCamp 126 raw TU interim files
df = load_dm_file("/mnt/dm_processed/cleaned/PSCamp126/J119565_D20220927_S488667_interimOutput_P001.dat.csv")#.select('PPPCL_CONSUMER_ID')
df = df.filter(df['PPPCL_CONSUMER_ID'].isin(ids))#.distinct()
df = append_external_data(df)
df = score_PIE(df)
df = score_HPE(df)
df = calculate_ndi(df)
df = score_LAE(df)
df = create_attributes_CP(df)
df = score_CM(df)
df = apply_tier_CPv6(df)
df = Part4_creating_reusable_features(df)
df = score_RR(df, rr_features_path=rr_features_path)
df = score_AR(df, ar_features_path=ar_features_path)
df = score_FR(df, fr_features_path=fr_features_path)
df = score_preUOL(df)

# COMMAND ----------

[col for col in df.columns if 'ndi' in col.lower()]

# COMMAND ----------

val_df = df.select('PPPCL_CONSUMER_ID',
                     'PIE_Income_Estimation','HPE_Housing_Payment_Estimation','LAE_Loan_Amount_Estimation','CM_Credit_Model_Score',
                     'PricingTier_v6','RR_ResponseProb','AR_ApprovalProb','FR_FundingProb','preUOLScore').toPandas()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Differences in RR

# COMMAND ----------

test = spark.read.csv("/mnt/science/Shringar/Databricks_Deployment/sample_126_w_ext_data_scored.csv", header=True, inferSchema=True)
rr_vars = list(pd.read_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/RR_v9_features.csv", header=None)[0])
test = test.select(*(['PPPCL_CONSUMER_ID','previous6Tag1','previous6Tag2','previous6Tag3','previous6Tag4','previous6Tag5','NDI',
                   'Tradeline_NDI','CPg_NDI','CPg_NdiWithLieDetector','CPg_LieDetectorDiff','CPg_RequestedLoanAmount','CPg_UnsecuredSummaryBalance','']+rr_vars)).toPandas()

rr_vars_new = list(pd.read_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/Version_Agnostic_Features/rrv9_version_agnostic_features.csv", header=None)[0])
df2 = df.select(*(['PPPCL_CONSUMER_ID','previous6Tag1','previous6Tag2','previous6Tag3','previous6Tag4','previous6Tag5','NDI',
                   'Tradeline_NDI','CPg_NDI','CPg_NdiWithLieDetector','CPg_LieDetectorDiff','CPg_RequestedLoanAmount','CPg_UnsecuredSummaryBalance']+rr_vars_new)).toPandas()

df2 = df2.merge(test, on='PPPCL_CONSUMER_ID', how='inner')

# COMMAND ----------

for col in df2.columns:
  if '_x' in col and 'previous' not in col:
    col = col.rsplit('_', 1)[0]
    print(f'{col}:', df2[['PPPCL_CONSUMER_ID',col + '_y',col + '_x']][~np.isclose(df2[col + '_y'], df2[col + '_x'], rtol=0)])

# COMMAND ----------

for col in ['previous6Tag1','previous6Tag2','previous6Tag3','previous6Tag4','previous6Tag5']:

# COMMAND ----------

np.isclose(df2['previous6Tag1_y'].astype('string'), df2['previous6Tag1_x'].astype('string'), rtol=0)

# COMMAND ----------

df2

# COMMAND ----------

df2[df2['PREVIOUS_COUNTER_5_y'].isnull() & df2['PREVIOUS_COUNTER_5_x'].notnull()][['previous6Tag1_y','previous6Tag2_y','previous6Tag3_y','previous6Tag4_y','previous6Tag5_y','previous6Tag1_x','previous6Tag2_x','previous6Tag3_x','previous6Tag4_x','previous6Tag5_x','PREVIOUS_COUNTER_5_y','PREVIOUS_COUNTER_5_x','RR_ResponseProb','ResponseProb_v9']]

# COMMAND ----------

df2.isnull().sum()

# COMMAND ----------

all(np.where(df2[df2['PREVIOUS_COUNTER_5_y'].notnull()]['PREVIOUS_COUNTER_5_y'] == df2[df2['PREVIOUS_COUNTER_5_x'].notnull()]['PREVIOUS_COUNTER_5_x'], True, False))

# COMMAND ----------

# MAGIC %md ### Differences in AR

# COMMAND ----------

test2 = spark.read.csv("/mnt/science/Shringar/Databricks_Deployment/sample_126_w_ext_data_scored.csv", header=True, inferSchema=True)
ar_vars = list(pd.read_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/ARv7_features.csv")['0'])
ar_vars = ['N05_S0_Y2' if col == 'N05_S0_Y2' else col for col in ar_vars]
# test2 = test2.select(*(['PPPCL_CONSUMER_ID']+ar_vars)).toPandas()

# ar_vars_new = list(pd.read_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/Version_Agnostic_Features/arv7_version_agnostic_features.csv", header=None)[0])
# df3 = df.select(*(['PPPCL_CONSUMER_ID']+ar_vars_new)).toPandas()

# df3 = df3.merge(test2, on='PPPCL_CONSUMER_ID', how='inner')

# COMMAND ----------

list(pd.read_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/ARv7_features.csv")['0'])
#pd.Series(ar.feature_names)#.to_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/ARv7_features.csv", index=False)

# COMMAND ----------

lae_mismatch = merged[['PPPCL_CONSUMER_ID','PredLoanAmount_v3','LAE_Loan_Amount_Estimation']][~np.isclose(merged['PredLoanAmount_v3'], merged['LAE_Loan_Amount_Estimation'], rtol=0)]
cm_mismatch = merged[['PPPCL_CONSUMER_ID','Model_v6_Score','CM_Credit_Model_Score']][~np.isclose(merged['Model_v6_Score'], merged['CM_Credit_Model_Score'], rtol=0)]
rr_mismatch = merged[['PPPCL_CONSUMER_ID','ResponseProb_v9','RR_ResponseProb']][~np.isclose(merged['ResponseProb_v9'], merged['RR_ResponseProb'], rtol=0)]
ar_mismatch = merged[['PPPCL_CONSUMER_ID','ApprovalProb_v7','AR_ApprovalProb']][~np.isclose(merged['ApprovalProb_v7'], merged['AR_ApprovalProb'], rtol=0)]
fr_mismatch = merged[['PPPCL_CONSUMER_ID','FundingProb_v7','FR_FundingProb']][~np.isclose(merged['FundingProb_v7'], merged['FR_FundingProb'], rtol=0)]
uol_mismatch = merged[['PPPCL_CONSUMER_ID','pre_UOL_score_v1','preUOLScore']][~np.isclose(merged['pre_UOL_score_v1'], merged['preUOLScore'], rtol=0)]

# COMMAND ----------

lae_mismatch

# COMMAND ----------

print(lae_mismatch.shape)
print(cm_mismatch.shape)
print(rr_mismatch.shape)
print(ar_mismatch.shape)
print(fr_mismatch.shape)
print(uol_mismatch.shape)

# COMMAND ----------

#df.write.csv("/mnt/science/Shringar/Databricks_Deployment/raw_files/new_sample_126.csv", header=True)

# COMMAND ----------


