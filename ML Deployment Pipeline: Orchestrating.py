# Databricks notebook source
import numpy as np
import datetime as dt
import joblib

import pandas as pd
import re
from pyspark.sql.functions import col, lit, when, desc
import pyspark.sql.functions as F

pd.set_option('display.max_rows', 500)

folder_path = "/dbfs/mnt/science/Shringar/Databricks_Deployment/"

# COMMAND ----------

# MAGIC %run ./ML_Deployment_Pipeline_Part_1

# COMMAND ----------

# MAGIC %run ./ML_Deployment_Pipeline_Part_2

# COMMAND ----------

# MAGIC %run ./ML_Deployment_Pipeline_Part_3

# COMMAND ----------

# MAGIC %run ./ML_Deployment_Pipeline_Part_4

# COMMAND ----------

all_input_fts = sorted(set(pie_features + lae_features + hpe_features + cm_features_clean + rr_fts + ar_fts + fr_fts))
len(all_input_fts)

# COMMAND ----------

scores = ['PPPCL_CONSUMER_ID','PIE_Income_Estimation','HPE_Housing_Payment_Estimation','LAE_Loan_Amount_Estimation',
          'CM_Credit_Model_Score','PricingTier_v6','RR_ResponseProb','AR_ApprovalProb','FR_FundingProb']

final_fts_skinny = ['PPPCL_CONSUMER_ID','NCOA_ADDR_State','GGENERC_gna001','CEMP08_b_finscr','V71_AT03S','V71_AT09S','V71_AT20S','V71_G069S','V71_G095S',
                    'V71_S114S','V71_RE101S','N05_G082','RECENT_TAG','PREVIOUS_COUNTER',
                    'PIE_Income_Estimation','LAE_Loan_Amount_Estimation', 'HPE_Housing_Payment_Estimation',
                    'DaysSinceOpenedUIL','MaxUILUtilization','UILAccountsOpenedPast12mo','RevolvingBalance','DaysSinceUILInquiry',
                    'Tradeline_NDI','UnsecuredSummaryBalance',
                    'LieDetectorRatio','MonthlyDti','Bti','PctNewBCTradesInPast12mo',
                    'CM_Credit_Model_Score', 'PricingTier_v6',
                    'RR_ResponseProb_v9','RR_ResponseProb',
                    'AR_ApprovalProb',
                    'FR_FundingProb',
                    'eligible_last_month','email_only','months_since_last_application','ehpm01_nopdfor', 'ehpm01_nopmod', 'ehpm01_nocpgt45','sendableemail',
                    'UnsecuredSummaryBalanceTUVars1', 'UnsecuredSummaryBalanceTUVars2','UnsecuredSummaryBalanceCustomVars', 'UnsecuredSummayBalanceBlend',
                    'lastreportedincome', 'lastreportedhousingpayment', 'bestcase_PIE_Income_Estimation_Or_Reported_Income_Blend', 'bestcase_MonthlyDti', 'bestcase_Bti', 
                    'bestcase_Tradeline_NDI','bestcase_CM_Credit_Model_Score','bestcase_PricingTier_v6']

scoring_inputs_reproducibility = ['NCOA_ADDR_ZipCode','W55_AGG911','V71_AT101S','V71_MT01S','V71_MT101S','V71_ST02S','V71_US02S','V71_US101S','RECENT_TAG',
                                  'V71_ST101S','V71_AU02S','V71_IN02S','V71_G960S','V71_IN09S','MeanIncome','MedianRental','mean_net_inv_income_tax','mean_state_sales_tax',
                                  'FederalTax','StateTax','MonthlyIncome','NDI','UILBalance','OpenUILAccounts','NonMortgageMonthlyPayment','LieDetectorDiff','LieDetectorHigh',
                                  'NdiWithLieDetector','BtiWithLieDetector','N05_S0Y2','HC03_VC113','no_student_trades','last_applied_date','last_partner_applied_date',
                                  'last_mailed_campaign_month','last_mailed_selection_month','lastreportedincome','lastreportedhousingpayment','last_mailed_campaign_month',
                                  'last_mailed_selection_month','ehpm01_nopnd','ehpm01_nocpay0','ehpm01_nocpay1','ehpm01_nocpgt46','V71_IN101S','V71_HR101S','V71_AU101S','epay01attr25',
                                  'epay01attr19']

final_fts_full = sorted(set(all_input_fts + final_fts_skinny + scoring_inputs_reproducibility))

# COMMAND ----------

## Scoring pipeline
file_path = "/mnt/dm_processed/cleaned/PSCamp131/"
df = load_dm_file(file_path, rename=True)
df = create_new_variables(df)
df = append_external_data(df)
df = score_PIE(df)
df = score_HPE(df)
df = calculate_ndi(df)
df = score_LAE(df)
df = create_attributes_CP(df)
df = score_CM(df)
df = apply_tier_CPv6(df)
df = Part4_creating_reusable_features(df)
df = score_RR(df, rr_features_path = rrv9_features_path, model_uri=f"models:/DM_RR_v9/Production", output_col = 'RR_ResponseProb_v9')
df = score_RR(df)
df = score_AR(df)
df = score_FR(df)

## Now running bestcase eligibility pipeline up until CM Model and pricing tier
df_bst = load_dm_file(file_path, rename=True).filter(col("lastreportedincome").isNotNull())
#print(df_bst.count())
df_bst = create_new_variables(df_bst)
df_bst = append_external_data(df_bst)
df_bst = score_PIE(df_bst)
df_bst = eligibility_bestcase_income(df_bst, output_col = "bestcase_PIE_Income_Estimation_Or_Reported_Income_Blend")
df_bst = score_HPE(df_bst)
df_bst = calculate_ndi(df_bst, income_name = "bestcase_PIE_Income_Estimation_Or_Reported_Income_Blend")
df_bst = score_LAE(df_bst)
df_bst = create_attributes_CP(df_bst, income_var = "bestcase_PIE_Income_Estimation_Or_Reported_Income_Blend")
df_bst = score_CM(df_bst)
df_bst = apply_tier_CPv6(df_bst)

df_bst = df_bst.select(*([col(col_name).alias("bestcase_" + col_name) for col_name in ['MonthlyDti','Bti','Tradeline_NDI','CM_Credit_Model_Score','PricingTier_v6']] +
                                                                                     ['PPPCL_CONSUMER_ID', 'bestcase_PIE_Income_Estimation_Or_Reported_Income_Blend']))

df = df.join(df_bst, on='PPPCL_CONSUMER_ID', how='left')

df2 = df.select(*final_fts_skinny)

#df2.write.mode('overwrite').csv("/mnt/dm_processed/scored/PSCamp130/dm_PSCamp130_scored_tiny", header=True)

# COMMAND ----------

df2.write.mode('overwrite').csv("/mnt/dm_processed/scored/PSCamp131/dm_PSCamp131_scored_tiny", header=True)
