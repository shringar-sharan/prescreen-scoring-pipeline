# Databricks notebook source
import numpy as np
import datetime as dt
import joblib

import pandas as pd
import re
from pyspark.sql.functions import col, lit
import pyspark.sql.functions as F

import mlflow
from pyspark.sql.functions import struct, col, when, broadcast

folder_path = "/dbfs/mnt/science/Shringar/Databricks_Deployment/"

# COMMAND ----------

state_tax_table_path = "/mnt/science/Shringar/Databricks_Deployment/StateTaxTable.csv"
cm_features_path = "/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/credit_model_v6_variables.csv"

interest_rate = 0.05
Ir = interest_rate/12

# COMMAND ----------

cm_features = list(pd.read_csv(cm_features_path, header=None)[0])
cm_features_clean = ['_'.join(col.split('_',2)[:2]) for col in cm_features]

# COMMAND ----------

def calculate_ndi_preprocessing(df):
    
  df = df.withColumn("V71_INAP01_Imputed", F.when(df.V71_INAP01.isNull() | (col("V71_INAP01") < 0),   107.90000).otherwise(col("V71_INAP01")))\
    .withColumn("V71_INAP01_Imputed", F.when( col("V71_INAP01") == -2,   2005.13000).otherwise(col("V71_INAP01_Imputed")))\
    .withColumn("V71_MT101S_Imputed", F.when( df.V71_MT101S.isNull() | (col("V71_MT101S") < 0),   -1.00000).otherwise(col("V71_MT101S")))\
    .withColumn("V71_MT21S_Imputed", F.when( df.V71_MT21S.isNull() | (col("V71_MT21S") < 0),   -1.00000).otherwise(col("V71_MT21S")))\
    .withColumn("V71_HIAP01_Imputed", F.when( df.V71_HIAP01.isNull() | (col("V71_HIAP01") < 0),   165.56700).otherwise(col("V71_HIAP01")))\
    .withColumn("V71_REAP01_Imputed", F.when( df.V71_REAP01.isNull()| (col("V71_REAP01") < 0), 245.27500).otherwise(col("V71_REAP01")))\
    .withColumn("V71_REAP01_Imputed", F.when(col("V71_REAP01") == -2,   133.25500).otherwise(col("V71_REAP01_Imputed")))\
    .withColumn("V71_S114S_Imputed", F.when( df.V71_S114S.isNull() | (col("V71_S114S") < 0), 1.00000).otherwise(col("V71_S114S")))\
    .withColumn("W49_ATTR06_Imputed", F.when( df.W49_ATTR06.isNull() | (col("W49_ATTR06") < 0), 92.98900).otherwise(col("W49_ATTR06")))\
    .withColumn("W49_ATTR10_Imputed", F.when( df.W49_ATTR10.isNull() | (col("W49_ATTR10") < 0), 0.00000).otherwise(col("W49_ATTR10")))\
    .withColumn("W49_AUP1003_Imputed", F.when( df.W49_AUP1003.isNull() | (col("W49_AUP1003") < 0), 1610.77000).otherwise(col("W49_AUP1003")))\
    #.withColumn("NDI", F.when(col("NDI") <= 500,500).otherwise(col("NDI")))\
    #.withColumn("NDI", F.when(col("NDI") >= 8000,8000).otherwise(col("NDI")))
  
  return df


def calculate_ndi(df, preprocess=True, state_tax_table_path = state_tax_table_path, 
                  income_name = "PIE_Income_Estimation", 
                  state_name = "NCOA_ADDR_State", 
                  housing_payment_name = "HPE_Housing_Payment_Estimation"):
  orig_cols = df.columns
  #print(orig_cols)
  
  df = (df.withColumn("PPPCL_CONSUMER_ID", col("PPPCL_CONSUMER_ID").cast('string'))
          .withColumn(state_name, col(state_name).cast('string'))
       )
  
  if preprocess:
    df = calculate_ndi_preprocessing(df)
  
  FedBracketsAmnt = [0,9075, 36900, 89350, 186350 , 405100 , 406750]
  FedBracketsAmntDiff = np.diff(FedBracketsAmnt).tolist()+[0]
  FedBracketsPercent = [0.10,0.15,0.25,0.28,0.33,0.35,0.396]
  
  df = (df.withColumn("Income", F.col(income_name)) # not_used_further
          .withColumn("V71_MT101S_Imputed", F.when(df['V71_MT101S_Imputed'].isNull(),0).otherwise(col("V71_MT101S_Imputed"))) # not_used_further
          .withColumn("V71_MT21S_Imputed",  F.when(df['V71_MT21S_Imputed'].isNull(),0).otherwise(col("V71_MT21S_Imputed"))) # not_used_further
          .withColumn("W49_ATTR10_Imputed", F.when(df['W49_ATTR10_Imputed'].isNull(),0).otherwise(col("W49_ATTR10_Imputed"))) # not_used_further
          .withColumn("MortAmnt", F.col("V71_MT101S_Imputed"))
          .withColumn("NumMortPay", F.col("V71_MT21S_Imputed"))
          .withColumn("NumMortPayMade", F.col("W49_ATTR10_Imputed"))
          .withColumn("NumMortPayRemaing", F.col("NumMortPayMade")-F.col("NumMortPay"))
          .withColumn("NumMortPayRemaing", F.when(col("NumMortPayRemaing") < 0, 360).otherwise(col("NumMortPayRemaing")))
          .withColumn("TotalMortInterest", F.col("MortAmnt")*(((Ir*F.col("NumMortPayRemaing")*(pow((1+Ir),F.col("NumMortPayRemaing"))))/(pow((1+Ir),F.col("NumMortPayRemaing"))-1))-1))
          .withColumn("YearAvgMortInterest", 12*F.col("TotalMortInterest")/F.col("NumMortPayRemaing"))
           )
  
  df = (df.withColumn("YearAvgMortInterest", F.when(col("YearAvgMortInterest").isNull(),0).otherwise(col("YearAvgMortInterest")))
          .withColumn("YearAvgMortInterest", F.when(col("YearAvgMortInterest") < 0, 0).otherwise(col("YearAvgMortInterest")))
          .withColumn("TaxableIncome", F.when(col("Income") < 258250, col("Income") - 4000).otherwise(col("Income")))
          .withColumn("TaxableIncome", F.col("TaxableIncome") - 6300)
          .withColumn("TaxableIncome", F.col("TaxableIncome") - F.col("YearAvgMortInterest"))
          .withColumn("federal_tax_brackets_1", F.when(col("TaxableIncome") > 9075, 9075).otherwise(col("TaxableIncome"))*0.10)
          .withColumn("federal_tax_brackets_2", F.when(col("TaxableIncome") > 36900, 27825).otherwise(col("TaxableIncome")-9075)*0.15)
          .withColumn("federal_tax_brackets_2", F.when(col("TaxableIncome") < 9075, 0).otherwise(col("federal_tax_brackets_2")))
          .withColumn("federal_tax_brackets_3", F.when(col("TaxableIncome") > 89350, 52450).otherwise(col("TaxableIncome")-36900)*0.25)
          .withColumn("federal_tax_brackets_3", F.when(col("TaxableIncome") < 36900, 0).otherwise(col("federal_tax_brackets_3")))
          .withColumn("federal_tax_brackets_4", F.when(col("TaxableIncome") > 186350, 97000).otherwise(col("TaxableIncome")-89350)*0.28)
          .withColumn("federal_tax_brackets_4", F.when(col("TaxableIncome") < 89350, 0).otherwise(col("federal_tax_brackets_4")))
          .withColumn("federal_tax_brackets_5", F.when(col("TaxableIncome") > 405100, 218750).otherwise(col("TaxableIncome")-186350)*0.33)
          .withColumn("federal_tax_brackets_5", F.when(col("TaxableIncome") < 186350, 0).otherwise(col("federal_tax_brackets_5")))
          .withColumn("federal_tax_brackets_6", F.when(col("TaxableIncome") > 406750, 1650).otherwise(col("TaxableIncome")-405100)*0.35)
          .withColumn("federal_tax_brackets_6", F.when(col("TaxableIncome") < 405100, 0).otherwise(col("federal_tax_brackets_6")))
          .withColumn("federal_tax_brackets_7", F.when(col("TaxableIncome") < 406750, 0).otherwise(col("TaxableIncome")-406750)*0.396)
          .withColumn("FederalTax", F.col("federal_tax_brackets_1") + F.col("federal_tax_brackets_2") + F.col("federal_tax_brackets_3") + 
                       F.col("federal_tax_brackets_4") + F.col("federal_tax_brackets_5") + F.col("federal_tax_brackets_6") + F.col("federal_tax_brackets_7"))
          .withColumn("FederalTax", F.when(col("FederalTax") < 0, 0).otherwise(col("FederalTax")))
           )
  
  SStaxThres = 113700
  SStaxRate = 0.062
  MedicRate = 0.0145
  StateTaxTable = spark.read.csv(state_tax_table_path, header=True, inferSchema=True)
  
  ##########################################################
  output = (df
           .withColumn("Income", F.col(income_name))
           .withColumn("State", F.col(state_name).cast('string'))
           )
  
  #print('Count1:', output.count())
  #print("before joining", output.count())
  
  output_temp = output.join(broadcast(StateTaxTable), on='State', how="left")#.drop(StateTaxTable.State)
  
  #print('Count2:', output_temp.count())
  #print(output_temp.select('PPPCL_CONSUMER_ID').distinct().count())
  
  output_temp = (output_temp.withColumn("AfterDeduction", F.col("Income")) #- F.col("Standard_Deduction") - F.col("Personal_Exemption"))
                            .withColumn("Money", F.col("AfterDeduction") - F.col("DollarsPreviousTaxed"))
                            .withColumn("Money", F.when(col("Money") > col("MaximumTaxableDollars"), col("MaximumTaxableDollars")).otherwise(col("Money")))
                            .withColumn("Money", F.when(col("Money") < 0, 0).otherwise(col("Money")))
                            .withColumn("StateTax", F.col("Money") * F.col("Rates"))
                )
  #print('Count3:', output_temp.count())
  #print(output_temp.select('PPPCL_CONSUMER_ID').distinct().count())
  
  output_temp = output_temp.groupBy("PPPCL_CONSUMER_ID").agg(F.max("AfterDeduction").alias("AfterDeduction_max"),
                                                         F.sum("StateTax").alias("StateTax_sum"),
                                                         F.sum("Money").alias("Money_sum"))
  
  #print('Count4:', output_temp.count())
  
  output = output.join(output_temp, on='PPPCL_CONSUMER_ID', how='inner')
  
  #print('Count5:', output.count())

  output = (output
            .withColumn("SSamt", F.when(col("AfterDeduction_max") >SStaxThres, SStaxThres*SStaxRate).otherwise(col("AfterDeduction_max")*SStaxRate))
            .withColumn("Medamt", F.col("Money_sum")*MedicRate)
            .withColumn("StateTax", F.col("StateTax_sum")+F.col("SSamt")+F.col("Medamt"))
            .withColumn("MonthlyIncome", (F.col(income_name)-F.col("FederalTax")-F.col("StateTax"))/12)
            .withColumn("TempStudent", F.when(output.W49_ATTR06_Imputed.isNull(),0).otherwise(col("W49_ATTR06_Imputed")))
            .withColumn("TempAuto", F.when(output.W49_AUP1003_Imputed.isNull(),0).otherwise(col("W49_AUP1003_Imputed")))
            .withColumn("NDI", F.col("MonthlyIncome")-F.col("TempStudent")-F.col("TempAuto")-F.col("V71_INAP01_Imputed")-F.col("V71_HIAP01_Imputed")-F.col("V71_REAP01_Imputed"))
            .withColumn("NDI", F.col("NDI")-F.col(housing_payment_name))
            .drop("TempStudent")
            .drop("TempAuto")
           )
  
  #print('Count6:', output.count())

  return output
  

# COMMAND ----------

#ndi_cols_used_further = ['MonthlyIncome','NDI']

# COMMAND ----------

# from pyspark.sql.functions import col, lit, udf, when, exp, pow

# def calculate_ndi(df, income_name = "PIE_v3", state_name = "NCOA_ADDR_State", housing_payment_name = "PHE_v2"):
#   FedBracketsAmnt = [0,9075, 36900, 89350, 186350 , 405100 , 406750]
#   FedBracketsAmntDiff = np.diff(FedBracketsAmnt).tolist()+[0]
#   FedBracketsPercent = [0.10,0.15,0.25,0.28,0.33,0.35,0.396]
#   interest_rate = 0.05
#   Ir = interest_rate/12
#   output = (df
#             .withColumn("Income", F.col(income_name)) # not_used_further
#             .withColumn("V71_MT101S_Imputed", F.when(df.V71_MT101S_Imputed.isNull(),0).otherwise(col("V71_MT101S_Imputed"))) # not_used_further
#             .withColumn("V71_MT21S_Imputed",  F.when(df.V71_MT21S_Imputed.isNull(),0).otherwise(col("V71_MT21S_Imputed"))) # not_used_further
#             .withColumn("W49_ATTR10_Imputed", F.when(df.W49_ATTR10_Imputed.isNull(),0).otherwise(col("W49_ATTR10_Imputed"))) # not_used_further
#             .withColumn("MortAmnt", F.col("V71_MT101S_Imputed"))
#             .withColumn("NumMortPay", F.col("V71_MT21S_Imputed"))
#             .withColumn("NumMortPayMade", F.col("W49_ATTR10_Imputed"))
#             .withColumn("NumMortPayRemaing", F.col("NumMortPayMade")-F.col("NumMortPay"))
#             .withColumn("NumMortPayRemaing", F.when(col("NumMortPayRemaing") < 0, 360).otherwise(col("NumMortPayRemaing")))
#             .withColumn("TotalMortInterest", F.col("MortAmnt")*(((Ir*F.col("NumMortPayRemaing")*(pow((1+Ir),F.col("NumMortPayRemaing"))))/(pow((1+Ir),F.col("NumMortPayRemaing"))-1))-1))
#             .withColumn("YearAvgMortInterest", 12*F.col("TotalMortInterest")/F.col("NumMortPayRemaing"))
#            )
#   output = (output
#             .withColumn("YearAvgMortInterest", F.when(output.YearAvgMortInterest.isNull(),0).otherwise(col("YearAvgMortInterest")))
#             .withColumn("YearAvgMortInterest", F.when(col("YearAvgMortInterest") < 0, 0).otherwise(col("YearAvgMortInterest")))
#             .withColumn("TaxableIncome", F.when(col("Income") < 258250, col("Income") - 4000).otherwise(col("Income")))
#             .withColumn("TaxableIncome", F.col("TaxableIncome") - 6300)
#             .withColumn("TaxableIncome", F.col("TaxableIncome") - F.col("YearAvgMortInterest"))
#             .withColumn("federal_tax_brackets_1", F.when(col("TaxableIncome") > 9075, 9075).otherwise(col("TaxableIncome"))*0.10)
#             .withColumn("federal_tax_brackets_2", F.when(col("TaxableIncome") > 36900, 27825).otherwise(col("TaxableIncome")-9075)*0.15)
#             .withColumn("federal_tax_brackets_2", F.when(col("TaxableIncome") < 9075, 0).otherwise(col("federal_tax_brackets_2")))
#             .withColumn("federal_tax_brackets_3", F.when(col("TaxableIncome") > 89350, 52450).otherwise(col("TaxableIncome")-36900)*0.25)
#             .withColumn("federal_tax_brackets_3", F.when(col("TaxableIncome") < 36900, 0).otherwise(col("federal_tax_brackets_3")))
#             .withColumn("federal_tax_brackets_4", F.when(col("TaxableIncome") > 186350, 97000).otherwise(col("TaxableIncome")-89350)*0.28)
#             .withColumn("federal_tax_brackets_4", F.when(col("TaxableIncome") < 89350, 0).otherwise(col("federal_tax_brackets_4")))
#             .withColumn("federal_tax_brackets_5", F.when(col("TaxableIncome") > 405100, 218750).otherwise(col("TaxableIncome")-186350)*0.33)
#             .withColumn("federal_tax_brackets_5", F.when(col("TaxableIncome") < 186350, 0).otherwise(col("federal_tax_brackets_5")))
#             .withColumn("federal_tax_brackets_6", F.when(col("TaxableIncome") > 406750, 1650).otherwise(col("TaxableIncome")-405100)*0.35)
#             .withColumn("federal_tax_brackets_6", F.when(col("TaxableIncome") < 405100, 0).otherwise(col("federal_tax_brackets_6")))
#             .withColumn("federal_tax_brackets_7", F.when(col("TaxableIncome") < 406750, 0).otherwise(col("TaxableIncome")-406750)*0.396)
#             .withColumn("FederalTax", F.col("federal_tax_brackets_1") + F.col("federal_tax_brackets_2") + F.col("federal_tax_brackets_3") + 
#                        F.col("federal_tax_brackets_4") + F.col("federal_tax_brackets_5") + F.col("federal_tax_brackets_6") + F.col("federal_tax_brackets_7"))
#             .withColumn("FederalTax", F.when(col("FederalTax") < 0, 0).otherwise(col("FederalTax")))
#            )

#   SStaxThres = 113700
#   SStaxRate = 0.062
#   MedicRate = 0.0145
#   StateTaxTable = spark.read.csv("/mnt/science/Chong/databricks/models/StateTaxTable.csv", header=True, inferSchema=True)
  
#   output = (output
#            .withColumn("Income", F.col(income_name))
#            .withColumn("State", F.col(state_name))
#            )
#   #print("before joining", output.count())
#   output_temp = output.join(StateTaxTable, output.State ==  StateTaxTable.State,"full").drop(StateTaxTable.State)
#   output_temp = (output_temp
#             .withColumn("AfterDeduction", F.col("Income")-F.col("Standard_Deduction")-F.col("Personal_Exemption"))
#             .withColumn("Money", F.col("AfterDeduction")-F.col("DollarsPreviousTaxed"))
#             .withColumn("Money", F.when(col("Money") > col("MaximumTaxableDollars"), col("MaximumTaxableDollars")).otherwise(col("Money")))
#             .withColumn("Money", F.when(col("Money") < 0, 0).otherwise(col("Money")))
#             .withColumn("StateTax", F.col("Money")*F.col("Rates"))
#            )
#   df_temp_AfterDeduction = output_temp.groupBy("PPPCL_CONSUMER_ID").max("AfterDeduction").withColumnRenamed( 'max(AfterDeduction)', "AfterDeduction_max")
#   df_temp_StateTax = output_temp.groupBy("PPPCL_CONSUMER_ID").sum("StateTax").withColumnRenamed( 'sum(StateTax)', "StateTax_sum")
#   df_temp_money = output_temp.groupBy("PPPCL_CONSUMER_ID").sum("Money").withColumnRenamed( 'sum(Money)', "Money_sum")

#   output = output.join(df_temp_AfterDeduction, output.PPPCL_CONSUMER_ID ==  df_temp_AfterDeduction.PPPCL_CONSUMER_ID,"inner").drop(df_temp_AfterDeduction.PPPCL_CONSUMER_ID)
#   output = output.join(df_temp_StateTax, output.PPPCL_CONSUMER_ID ==  df_temp_StateTax.PPPCL_CONSUMER_ID,"inner").drop(df_temp_StateTax.PPPCL_CONSUMER_ID)
#   output = output.join(df_temp_money, output.PPPCL_CONSUMER_ID ==  df_temp_money.PPPCL_CONSUMER_ID,"inner").drop(df_temp_money.PPPCL_CONSUMER_ID)

#   output = (output
#             .withColumn("SSamt", F.when(col("AfterDeduction_max") >SStaxThres, SStaxThres*SStaxRate).otherwise(col("AfterDeduction_max")*SStaxRate))
#             .withColumn("Medamt", F.col("Money_sum")*MedicRate)
#             .withColumn("StateTax", F.col("StateTax_sum")+F.col("SSamt")+F.col("Medamt"))
#             .withColumn("MonthlyIncome", (F.col(income_name)-F.col("FederalTax")-F.col("StateTax"))/12)
#             .withColumn("TempStudent", F.when(output.W49_ATTR06_Imputed.isNull(),0).otherwise(col("W49_ATTR06_Imputed")))
#             .withColumn("TempAuto", F.when(output.W49_AUP1003_Imputed.isNull(),0).otherwise(col("W49_AUP1003_Imputed")))
#             .withColumn("NDI", F.col("MonthlyIncome")-F.col("TempStudent")-F.col("TempAuto")-F.col("V71_INAP01_Imputed")-F.col("V71_HIAP01_Imputed")-F.col("V71_REAP01_Imputed"))
#             .withColumn("NDI", F.col("NDI")-F.col(housing_payment_name))
#             .drop("TempStudent")
#             .drop("TempAuto")
#            )

#   return output

# COMMAND ----------

#tradeline_ndi_cols_used_further = ['MonthlyIncome','Tradeline_NDI']

# COMMAND ----------

def rename_attributes_cp(df, housing_estimate = "HPE_Housing_Payment_Estimation",                       income_estimate = "PIE_Income_Estimation", 
                         predicted_loan_amount = "LAE_Loan_Amount_Estimation",    state = "NCOA_ADDR_State",
                         revolving_balance_varname = "epay01attr25",            uil_balance_varname = "epay01attr19", 
                         non_mortgage_monthly_payment_varname = "epay01attr24", max_uil_utilization_varname = "epay01attr21", 
                         days_since_opened_uil_varname = "epay01attr20",        days_since_uil_inquiry_varname = "epay01attr26", 
                         uil_accounts_opened_past_12mo_varname = "epay01attr23",open_uil_accounts_varname = "epay01attr22", FICO = "CEMP08_b_finscr"):
  output = (df
            .withColumn("CPg_HousingPayment", F.col(housing_estimate))
            .withColumn("CPg_Income", F.col(income_estimate))
            .withColumn("CPg_RequestedLoanAmount", F.col(predicted_loan_amount))
            .withColumn("CPg_State", F.col(state))
            .withColumn("CPg_RevolvingBalance", F.col(revolving_balance_varname))
            .withColumn("CPg_UILBalance", F.col(uil_balance_varname))
            .withColumn("CPg_NonMortgageMonthlyPayment", F.col(non_mortgage_monthly_payment_varname))
            .withColumn("CPg_MaxUILUtilization", F.col(max_uil_utilization_varname)) # reqd for eligibility
            .withColumn("CPg_DaysSinceOpenedUIL", F.col(days_since_opened_uil_varname))
            .withColumn("CPg_DaysSinceUILInquiry", F.col(days_since_uil_inquiry_varname))
            .withColumn("CPg_UILAccountsOpenedPast12mo", F.col(uil_accounts_opened_past_12mo_varname)) # reqd for eligibility
            .withColumn("CPg_OpenUILAccounts", F.col(open_uil_accounts_varname))    
            .withColumn("CPg_FicoScore", F.col(FICO)) 
           )
  return output
###########################################################################################################################
def calculate_tradeline_ndi(df, income_name = "PIE_Income_Estimation", state_name = "NCOA_ADDR_State", housing_payment_name = "HPE_Housing_Payment_Estimation"):
  output = (df
            .withColumn("MonthlyIncome", (F.col(income_name)-F.col("FederalTax")-F.col("StateTax"))/12)
            .withColumn("TempRevolving", F.when(col("V71_REAP01").isNull(),0).otherwise(col("V71_REAP01")))
            .withColumn("TempInstallment", F.when(col("V71_INAP01").isNull(),0).otherwise(col("V71_INAP01")))
            .withColumn("Tradeline_NDI", F.col("MonthlyIncome")-F.col("TempRevolving")-F.col("TempInstallment"))
            .withColumn("Tradeline_NDI", F.col("Tradeline_NDI")-F.col(housing_payment_name))
            .withColumn("CPg_NDI", F.col("Tradeline_NDI"))
            .drop("TempRevolving")
            .drop("TempInstallment")
    )
  return output
###########################################################################################################################
def treat_attributes_derived_attribute_input_CP(df):
  output = (df
            .withColumn("CPv5_5_SNC_RevolvingBalance", F.when(F.col("CPg_RevolvingBalance") < 0, 0).otherwise(col("CPg_RevolvingBalance")))
            .withColumn("CPv5_5_SNC_UILBalance", F.when(F.col("CPg_UILBalance") < 0, 0).otherwise(col("CPg_UILBalance")))
            .withColumn("CPv5_5_SNC_NonMortgageMonthlyPayment", F.when(F.col("CPg_NonMortgageMonthlyPayment") < 0, 0).otherwise(col("CPg_NonMortgageMonthlyPayment")))
            .withColumn("CPv5_5_SNC_V71_BC03S", F.when(F.col("V71_BC03S") <= 0, -1).otherwise(col("V71_BC03S")))
            .withColumn("CPv5_5_SNC_V71_BC25S", F.when(F.col("V71_BC25S") <= 0, -1).otherwise(col("V71_BC25S")))
            .withColumn("CPv5_5_SNC_W55_TRV01", F.when(F.col("W55_TRV01") <= 1, 1).otherwise(col("W55_TRV01")))
            .withColumn("CPv5_5_SNC_W55_TRV02", F.when(F.col("W55_TRV02") <= -1, 0).otherwise(col("W55_TRV02")))
            .withColumn("CPv5_5_SNC_W55_TRV02", F.when(F.col("CPv5_5_SNC_W55_TRV02") > 12, 12).otherwise(col("CPv5_5_SNC_W55_TRV02")))
    )
  return output
###########################################################################################################################
def create_derived_attributes_CP(df):
  df = (df
          .withColumn("CPg_UnsecuredSummaryBalance",  (F.col("CPv5_5_SNC_RevolvingBalance") + F.col("CPv5_5_SNC_UILBalance")))
          .withColumn("CPg_LieDetectorRatio",         (F.when(F.col("CPg_UnsecuredSummaryBalance")!=0, F.col("CPg_RequestedLoanAmount") / F.col("CPg_UnsecuredSummaryBalance")).otherwise(np.inf)))
          .withColumn("CPg_LieDetectorDiff",          (F.col("CPg_RequestedLoanAmount") - F.col("CPg_UnsecuredSummaryBalance")))
       )
  output = (df   
            
        ###########################################################################################################################
        #  .withColumn("CPg_LieDetectorRatio",         F.when(F.col("CPg_UnsecuredSummaryBalance")==0, np.inf))
        ###########################################################################################################################
          .withColumn("CPg_LieDetectorHigh",           F.when(((col("CPg_LieDetectorRatio").isNotNull()) & (col("CPg_LieDetectorRatio")>= 1)),
                                                              1-(col("CPg_UnsecuredSummaryBalance")/col("CPg_RequestedLoanAmount"))).otherwise(0))
          .withColumn("CPg_NdiWithLieDetector",       (F.col("CPg_Ndi") - (F.when(F.col("CPg_LieDetectorDiff")> 0, F.col("CPg_LieDetectorDiff")).otherwise(0))* 0.01744269))
          .withColumn("CPg_MonthlyDti",               (F.col("CPv5_5_SNC_NonMortgageMonthlyPayment") + F.col("CPg_HousingPayment")) / F.col("CPg_Income") * 12) # reqd for eligibility
          .withColumn("CPg_Bti",                       F.col("CPg_UnsecuredSummaryBalance") / F.col("CPg_Income"))
          .withColumn("CPg_BtiWithLieDetector",       (F.col("CPg_Bti") + (F.when(F.col("CPg_LieDetectorDiff")> 0, F.col("CPg_LieDetectorDiff")).otherwise(0))/ F.col("CPg_Income")))
          .withColumn("CPg_PctNewBCTradesInPast12mo", (F.when( (1- F.col("CPv5_5_SNC_V71_BC25S")/F.col("CPv5_5_SNC_V71_BC03S"))> 1, 
                                                              1).otherwise(1 - F.col("CPv5_5_SNC_V71_BC25S")/F.col("CPv5_5_SNC_V71_BC03S")))) # reqd for eligibility
          .withColumn("CPg_OverlimitTrend",            F.col("CPv5_5_SNC_W55_TRV02") / F.col("CPv5_5_SNC_W55_TRV01"))
          .withColumn("CPg_W55_AGG910_UNF",            F.when(F.col("W55_AGG910") < 0, 1).otherwise(0))
            
            .withColumn("PayoffBti",                 F.col("CPg_Bti"))
            .withColumn("Bti",                       F.col("CPg_Bti"))
            .withColumn("PayoffDtiWithLieDetector",  F.col("CPg_BtiWithLieDetector"))
            .withColumn("BtiWithLieDetector",        F.col("CPg_BtiWithLieDetector"))
            .withColumn("IndustryDti",               F.col("CPg_MonthlyDti"))
            .withColumn("MonthlyDti",                F.col("CPg_MonthlyDti"))    # reqd for eligibility
            .withColumn("NdiWithLieDetector",        F.col("CPg_NdiWithLieDetector"))
            .withColumn("UILBalance",                F.col("CPg_UILBalance"))
            .withColumn("UnsecuredSummaryBalance",   F.col("CPg_UnsecuredSummaryBalance"))
            .withColumn("NonMortgageMonthlyPayment", F.col("CPg_NonMortgageMonthlyPayment"))
            .withColumn("UILAccountsOpenedPast12mo", F.col("CPg_UILAccountsOpenedPast12mo")) # reqd for eligibility
            .withColumn("DaysSinceOpenedUIL",        F.col("CPg_DaysSinceOpenedUIL"))
            .withColumn("DaysSinceUILInquiry",       F.col("CPg_DaysSinceUILInquiry"))
            .withColumn("MaxUILUtilization",         F.col("CPg_MaxUILUtilization")) # reqd for eligibility
            .withColumn("OpenUILAccounts",           F.col("CPg_OpenUILAccounts"))
            .withColumn("RevolvingBalance",          F.col("CPg_RevolvingBalance"))
            .withColumn("LieDetectorRatio",          F.col("CPg_LieDetectorRatio"))
            .withColumn("LieDetectorDiff",           F.col("CPg_LieDetectorDiff"))
            .withColumn("LieDetectorHigh",           F.col("CPg_LieDetectorHigh"))
            .withColumn("PctNewBCTradesInPast12mo",  F.col("CPg_PctNewBCTradesInPast12mo")) # reqd for eligibility

           )
  return output
###########################################################################################################################
def create_attributes_CP(df, income_var = "PIE_Income_Estimation"):
  output = rename_attributes_cp(df, income_estimate = income_var)
  output = calculate_tradeline_ndi(output, income_name = income_var)
  output = treat_attributes_derived_attribute_input_CP(output)
  output = create_derived_attributes_CP(output)
  return output


# COMMAND ----------

# features = list(pd.read_csv("/dbfs/mnt/science/Shringar/Databricks_Deployment/Features/credit_model_v6_variables.csv", header=None)[0])
# features_clean = ['_'.join(col.split('_',2)[:2]) for col in features]
# features_clean

# COMMAND ----------

# # Original scoring function (version agnostic)
# def score_CM(df, preprocess=True, cm_features_path = cm_features_path, model_uri="models:/CM_v6/1"):
  
#   if preprocess:
#     df = treat_attributes_credit_model_CPv6(df)

#   features = list(pd.read_csv(cm_features_path, header=None)[0])
#   features_clean = ['_'.join(col.split('_',2)[:2]) for col in features]

#   model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
#   df = df.withColumn("CM_Credit_Model_Score", model(struct(*map(col, features_clean))))
  
#   #df = apply_tier_CPv6(df)
#   #output_sub = output.select("PPPCL_CONSUMER_ID", "Model_v6_Score", "PricingTier_v6")
#   #output = df.join(output_sub, df.PPPCL_CONSUMER_ID ==  output_sub.PPPCL_CONSUMER_ID,"inner").drop(output_sub.PPPCL_CONSUMER_ID) # COULD BE TIME CONSUMING
#   return df
  

# def treat_attributes_credit_model_CPv6(df):
  
#   # Preprocess CM
#   df = (df
#             .withColumn("W55_TRV07", F.when((df.W55_TRV07.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV07")))
#             .withColumn("W55_TRV08", F.when((df.W55_TRV08.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV08")))
#             .withColumn("W55_TRV09", F.when((df.W55_TRV09.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV09")))
#             .withColumn("W55_TRV10", F.when((df.W55_TRV10.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV10")))
            
#             .withColumn("W55_AGG909", F.when((df.W55_AGG909.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_AGG909")))
#             .withColumn("W55_TRV17", F.when((df.W55_TRV17.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV17")))
            
#             .withColumn("V71_AT36S", F.when((df.V71_AT36S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_AT36S")))
#             .withColumn("V71_AU36S", F.when((df.V71_AU36S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_AU36S")))
#             .withColumn("V71_IN36S", F.when((df.V71_IN36S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_IN36S")))
#             .withColumn("V71_MT36S", F.when((df.V71_MT36S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_MT36S")))
#             .withColumn("V71_ST36S", F.when((df.V71_ST36S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_ST36S")))
#             .withColumn("W55_TRV01", F.when((df.W55_TRV01.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_TRV01")))
            
#             .withColumn("W55_PAYMNT08", F.when((df.W55_PAYMNT08.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("W55_PAYMNT08")))
#             .withColumn("V71_AT21S", F.when((df.V71_AT21S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_AT21S")))
#             .withColumn("V71_BC21S", F.when((df.V71_BC21S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_BC21S")))
#             .withColumn("V71_BR21S", F.when((df.V71_BR21S.isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_BR21S")))
            
#             .withColumn("DaysSinceMostRecentUILInquiry",            F.when((df.epay01attr26.isin(-6, -5, -4, -3, -2, -1)), 1e+05 ).otherwise(F.col("epay01attr26")))
#             .withColumn("MaxUtilizationOfUnsecuredInstallmentLoan", F.when((df.epay01attr21.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr21")))
#             .withColumn("UnsecuredInstallmentLoanBalance",          F.when((df.epay01attr19.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr19")))
#             .withColumn("UnsecuredInstallmentLoanTrades",           F.when((df.epay01attr22.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr22")))
#             .withColumn("UnsecuredInstallmentTradesOpenedLast12mo", F.when((df.epay01attr23.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr23")))
# #             .withColumn("V71_AT104S_pct_all_trd_opned_pst_24mo_all_trd",                      F.col("V71_AT104S"))
# #             .withColumn("V71_AT31S_pct_opn_trd_gt_75pct_credln_verif_pst_12mo",               F.col("V71_AT31S"))
# #             .withColumn("V71_AT34B_util_for_opn_trd_verif_pst_12mo_excl_mtg_home_equity",     F.col("V71_AT34B"))
# #             .withColumn("V71_AT36S_mo_since_most_recent_dq",                                  F.col("V71_AT36S"))
# #             .withColumn("V71_BC09S_num_cc_trd_opned_pst_24mo",                                F.col("V71_BC09S"))
# #             .withColumn("V71_BC102S_avg_credln_opn_cc_trd_verif_pst_12mo",                    F.col("V71_BC102S"))
# #             .withColumn("V71_BC104S_avg_the_opn_cc_trd_utils_verif_pst_12mo",                 F.col("V71_BC104S"))
# #             .withColumn("V71_BC20S_mo_since_oldest_cc_trd_opned",                             F.col("V71_BC20S"))
# #             .withColumn("V71_BC21S_mo_since_most_recent_cc_trd_opned",                        F.col("V71_BC21S"))
# #             .withColumn("V71_BC31S_pct_opn_cc_trd_gt_75pct_credln_verif_pst_12mo",            F.col("V71_BC31S"))
# #             .withColumn("V71_BC97A_total_opn_buy_opn_cc_verif_pst_3mo",                       F.col("V71_BC97A"))
# #             .withColumn("V71_BC98A_total_opn_buy_opn_cc_verif_pst_12mo",                      F.col("V71_BC98A"))
# #             .withColumn("V71_BR20S_mo_since_oldest_bank_rvl_trd_opned",                       F.col("V71_BR20S"))
# #             .withColumn("V71_BR31S_pct_opn_bank_rvl_trd_gt_75pct_credln_verif_pst_12mo",      F.col("V71_BR31S"))
# #             .withColumn("V71_FI34S_util_for_opn_fin_instlmnt_trd_verif_pst_12mo",             F.col("V71_FI34S"))
# #             .withColumn("V71_G001B_num_30_or_more_dpd_ratings_pst_12mo",                      F.col("V71_G001B"))
# #             .withColumn("V71_G001S_num_30dpd_ratings_pst_12mo",                               F.col("V71_G001S"))
# #             .withColumn("V71_G201A_total_opn_buy_opn_trd_verif_pst_3mo_excl_instlmnt_mtg",    F.col("V71_G201A"))
# #             .withColumn("V71_G202A_total_opn_buy_opn_trd_verif_pst_12mo_excl_instlmnt_mtg",   F.col("V71_G202A"))
# #             .withColumn("V71_G242F_num_fin_inq_includes_dup_pst_3mo",                         F.col("V71_G242F"))
# #             .withColumn("V71_G242S_num_inq_includes_dup_pst_3mo",                             F.col("V71_G242S"))
# #             .withColumn("V71_G243F_num_fin_inq_includes_dup_pst_6mo",                         F.col("V71_G243F"))
# #             .withColumn("V71_G244F_num_fin_inq_includes_dup_pst_12mo",                        F.col("V71_G244F"))
# #             .withColumn("V71_G250B_num_30dpd_or_worse_itm_pst_12mo_excl_med_collect_itm",     F.col("V71_G250B"))
# #             .withColumn("V71_G250C_num_30dpd_or_worse_itm_pst_24mo_excl_med_collect_itm",     F.col("V71_G250C"))
# #             .withColumn("V71_G960S_num_dedup_inq",                                            F.col("V71_G960S"))
# #             .withColumn("V71_G980S_num_dedup_inq_pst_6mo",                                    F.col("V71_G980S"))
# #             .withColumn("V71_G990S_num_dedup_inq_pst_12mo",                                   F.col("V71_G990S"))
# #             .withColumn("V71_IN36S_mo_since_most_recent_instlmnt_dq",                         F.col("V71_IN36S"))
# #             .withColumn("V71_RE102S_avg_credln_opn_rvl_trd_verif_pst_12mo",                   F.col("V71_RE102S"))
# #             .withColumn("V71_RE31S_pct_opn_rvl_trd_gt_75pct_credln_verif_pst_12mo",           F.col("V71_RE31S"))
# #             .withColumn("V71_S004S_avg_num_mo_trd_have_been_on_file",                         F.col("V71_S004S")) 
# #             .withColumn("V71_S114S_num_dedup_inq_pst_6mo_excl_auto_mtg_inq",                  F.col("V71_S114S")) # reqd for eligibility
# #             .withColumn("V71_S204S_total_bal_third_party_collect_verif_pst_12mo",             F.col("V71_S204S"))
# #             .withColumn("W55_AGG909_mo_since_max_agg_bnkcrd_bal_over_last_12mo",              F.col("W55_AGG909"))
# #             .withColumn("W55_AGG910_max_agg_bnkcrd_util_over_last_3mo",                       F.col("W55_AGG910"))
# #             .withColumn("W55_AGGS904_peak_mo_bnkcrd_spend_over_pst_12mo",                     F.col("W55_AGGS904"))
# #             .withColumn("W55_BALMAG01_non_mtg_bal_magnitude",                                 F.col("W55_BALMAG01"))
# #             .withColumn("W55_BALMAG02_rvl_bal_magnitude",                                     F.col("W55_BALMAG02"))
# #             .withColumn("W55_INDEX01_annual_yoy_spend_index",                                 F.col("W55_INDEX01"))
# #             .withColumn("W55_INDEX02_most_recent_quarter_yoy_spend_index",                    F.col("W55_INDEX02"))
# #             .withColumn("W55_PAYMNT08_ratio_actual_min_pmt_for_rvl_trd_last_mo",              F.col("W55_PAYMNT08"))
# #             .withColumn("W55_PAYMNT10_num_pmt_last_3mo",                                      F.col("W55_PAYMNT10"))
# #             .withColumn("W55_PAYMNT11_num_pmt_last_12mo",                                     F.col("W55_PAYMNT11"))
# #             .withColumn("W55_REVS904_max_agg_rvl_mo_spend_over_last_12mo",                    F.col("W55_REVS904"))
# #             .withColumn("W55_RVDEX01_annual_yoy_rvl_spend_index",                             F.col("W55_RVDEX01"))
# #             .withColumn("W55_RVDEX02_most_recent_quarter_yoy_rvl_spend_index",                F.col("W55_RVDEX02"))
# #             .withColumn("W55_RVLR01_util_for_bnkcrd_acct_with_a_rvl_bal",                     F.col("W55_RVLR01"))
# #             .withColumn("W55_TRV01_num_mo_since_overlimit_on_a_bnkcrd",                       F.col("W55_TRV01"))
# #             .withColumn("W55_TRV02_num_mo_overlimit_on_a_bnkcrd_over_last_12mo",              F.col("W55_TRV02"))
# #             .withColumn("W55_TRV03_num_non_mtg_trd_with_a_bal_incr_last_mo",                  F.col("W55_TRV03"))
# #             .withColumn("W55_TRV04_num_non_mtg_bal_incr_last_3mo",                            F.col("W55_TRV04"))
# #             .withColumn("W55_TRV07_num_non_mtg_bal_decr_last_mo",                             F.col("W55_TRV07"))
# #             .withColumn("W55_TRV08_num_non_mtg_bal_decr_last_3mo",                            F.col("W55_TRV08"))
# #             .withColumn("W55_TRV09_num_non_mtg_bal_decr_yoy",                                 F.col("W55_TRV09"))
# #             .withColumn("W55_TRV10_num_mo_non_mtg_bal_decr_last_12mo",                        F.col("W55_TRV10"))
# #             .withColumn("W55_TRV11_num_rvl_high_cred_incr_last_mo",                           F.col("W55_TRV11"))
# #             .withColumn("W55_TRV12_num_rvl_high_cred_incr_last_3mo",                          F.col("W55_TRV12"))
# #             .withColumn("W55_TRV13_num_rvl_high_cred_incr_yoy",                               F.col("W55_TRV13"))
# #             .withColumn("W55_TRV22_num_mo_bnkcrd_cred_limit_decr_last_12mo",                  F.col("W55_TRV22"))
# #             .withColumn("V71_AT06S_num_trd_opned_pst_6mo",                                    F.col("V71_AT06S"))
# #             .withColumn("V71_AT09S_num_trd_opned_pst_24mo",                                   F.col("V71_AT09S"))
# #             .withColumn("V71_AT21S_mo_since_most_recent_trd_opned",                           F.col("V71_AT21S"))
# #             .withColumn("V71_AT28A_total_credln_opn_trd_verif_pst_12mo",                      F.col("V71_AT28A"))
# #             .withColumn("V71_AT30S_pct_opn_trd_more_than_50pct_credln_verif_pst_12mo",        F.col("V71_AT30S"))
# #             .withColumn("V71_AT32S_max_bal_owed_on_opn_trd_verif_pst_12mo",                   F.col("V71_AT32S"))
# #             .withColumn("V71_AU36S_mo_since_most_recent_auto_dq",                             F.col("V71_AU36S"))       
# #             .withColumn("V71_BC34S_util_for_opn_cc_trd_verif_pst_12mo",                       F.col("V71_BC34S"))
# #             .withColumn("V71_BC35S_avg_bal_opn_cc_trd_verif_pst_12mo",                        F.col("V71_BC35S"))
# #             .withColumn("V71_BR09S_num_bank_rvl_trd_opned_pst_24mo",                          F.col("V71_BR09S"))
# #             .withColumn("V71_BR21S_mo_since_most_recent_bank_rvl_trd_opned",                  F.col("V71_BR21S"))
# #             .withColumn("V71_FI06S_num_fin_instlmnt_trd_opned_pst_6mo",                       F.col("V71_FI06S"))
# #             .withColumn("V71_FI09S_num_fin_instlmnt_trd_opned_pst_24mo",                      F.col("V71_FI09S"))
# #             .withColumn("V71_FI30S_pct_opn_fin_instlmnt_trd_gt_50pct_credln_verif_pst_12mo",  F.col("V71_FI30S"))
# #             .withColumn("V71_FI31S_pct_opn_fin_instlmnt_trd_gt_75pct_credln_verif_pst_12mo",  F.col("V71_FI31S"))
# #             .withColumn("V71_G058S_num_trd_30_or_more_dpd_pst_6mo",                           F.col("V71_G058S"))
# #             .withColumn("V71_G059S_num_trd_30_or_more_dpd_pst_12mo",                          F.col("V71_G059S"))
# #             .withColumn("V71_G061S_num_trd_30_or_more_dpd_pst_24mo",                          F.col("V71_G061S"))
# #             .withColumn("V71_G213A_highest_bal_third_party_collect_verif_24mo",               F.col("V71_G213A"))
# #             .withColumn("V71_G213B_highest_bal_non_med_third_party_collect_verif_24mo",       F.col("V71_G213B"))
# #             .withColumn("V71_G215A_num_third_party_collect_with_bal_larger_than_0_dollar",    F.col("V71_G215A"))
# #             .withColumn("V71_G234S_num_day_with_inquiry_occurring_pst_30day",                 F.col("V71_G234S"))
# #             .withColumn("V71_G237S_num_inq_pst_6mo_includes_dup",                             F.col("V71_G237S"))
# #             .withColumn("V71_G238S_num_inq_pst_12mo_includes_dup",                            F.col("V71_G238S"))
# #             .withColumn("V71_G244S_num_inq_pst_12mo_includes_dup",                            F.col("V71_G244S"))
# #             .withColumn("V71_G251A_num_60dpd_or_worse_itm_ever_excl_med_collect_itm",         F.col("V71_G251A"))
# #             .withColumn("V71_G251B_num_60dpd_or_worse_itm_pst_12mo_excl_med_collect_itm",     F.col("V71_G251B"))
# #             .withColumn("V71_G310S_worst_rating_on_all_trd_pst_12mo",                         F.col("V71_G310S"))
# #             .withColumn("V71_MT20S_mo_since_oldest_mtg_trd_opned",                            F.col("V71_MT20S"))
# #             .withColumn("V71_MT28S_total_credln_opn_mtg_trd_verif_pst_12mo",                  F.col("V71_MT28S"))
# #             .withColumn("V71_MT36S_mo_since_most_recent_mtg_dq",                              F.col("V71_MT36S"))
# #             .withColumn("V71_PB20S_mo_since_oldest_premium_cc_trd_opned",                     F.col("V71_PB20S"))
# #             .withColumn("V71_RE09S_num_rvl_trd_opned_pst_24mo",                               F.col("V71_RE09S"))
# #             .withColumn("V71_RE20S_mo_since_oldest_rvl_trd_opned",                            F.col("V71_RE20S"))
# #             .withColumn("V71_RE28S_total_credln_opn_rvl_trd_verif_pst_12mo",                  F.col("V71_RE28S"))
# #             .withColumn("V71_RE30S_pct_opn_rvl_trd_gt_50pct_credln_verif_pst_12mo",           F.col("V71_RE30S"))
# #             .withColumn("V71_RE34S_util_for_opn_rvl_trd_verif_pst_12mo",                      F.col("V71_RE34S"))
# #             .withColumn("V71_RT31S_pct_opn_retail_trd_gt_75pct_credln_verif_pst_12mo",        F.col("V71_RT31S"))
# #             .withColumn("V71_S043S_num_opn_trd_gt_50pct_credln_verif_pst_12mo_excl_instlmnt_mtg",F.col("V71_S043S"))
# #             .withColumn("V71_ST36S_mo_since_most_recent_student_ln_dq",                       F.col("V71_ST36S"))
# #             .withColumn("V71_ST99S_total_bal_all_student_ln_trd_ever_dq",                     F.col("V71_ST99S"))
# #             .withColumn("W55_AGG904_num_agg_non_mtg_cred_limit_decr_over_last_quarter",       F.col("W55_AGG904"))
# #             .withColumn("W55_AGG911_max_agg_bnkcrd_util_over_last_year",                      F.col("W55_AGG911"))
# #             .withColumn("W55_REVS901_agg_rvl_mo_spend_over_last_3mo",                         F.col("W55_REVS901"))
# #             .withColumn("W55_TRV17_num_bnkcrd_acct_with_a_yoy_cred_limit_incr",               F.col("W55_TRV17"))
# #             .withColumn("W55_TRV21_num_bnkcrd_acct_with_a_yoy_cred_limit_decr",               F.col("W55_TRV21"))
# #             .withColumn("V71_AT20S_mo_since_oldest_trd_opn",                                  F.col("V71_AT20S")) # reqd for eligibility
            
#     )
#   return df
# ###########################################################################################################################
# def apply_tier_CPv6(df, cm_score_var="CM_Credit_Model_Score"):
#   output = (df
#             .withColumn("PricingTier_v6", when(col(cm_score_var) < 0.047079, 1)
#                         .when(col(cm_score_var) < 0.1424,   2)
#                         .when(col(cm_score_var) < 0.251362, 3)
#                         .when(col(cm_score_var) < 0.330307, 4)
#                         .when(col(cm_score_var) < 0.367165, 5)
#                         .otherwise(6))
#            )
#   return output

# COMMAND ----------

def score_CM(df, cm_features_path = cm_features_path, model_uri="models:/DM_CM_v6/1", output_col = 'CM_Credit_Model_Score'):
  
  features = list(pd.read_csv(cm_features_path, header=None)[0])
  features_clean = ['_'.join(col.split('_',2)[:2]) for col in features]
  CM_cols = list(set(features_clean).intersection(set(df.columns))) + [col for col in df.columns if 'epay' in col]
  
  # Create new variables
  
  df = (df.withColumn("DaysSinceMostRecentUILInquiry", F.when((df.epay01attr26.isin(-6, -5, -4, -3, -2, -1)), 1e+05 ).otherwise(F.col("epay01attr26")))
          .withColumn("MaxUtilizationOfUnsecuredInstallmentLoan", F.when((df.epay01attr21.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr21")))
          .withColumn("UnsecuredInstallmentLoanBalance", F.when((df.epay01attr19.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr19")))
          .withColumn("UnsecuredInstallmentLoanTrades", F.when((df.epay01attr22.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr22")))
          .withColumn("UnsecuredInstallmentTradesOpenedLast12mo", F.when((df.epay01attr23.isin(-6, -5, -4, -3, -2, -1)), 0 ).otherwise(F.col("epay01attr23")))
       )
  
#   df['DaysSinceMostRecentUILInquiry'] = np.where(df['epay01attr26'].isin([-6, -5, -4, -3, -2, -1]), 1e+05, df['epay01attr26'])
#   df['MaxUtilizationOfUnsecuredInstallmentLoan'] = np.where(df['epay01attr21'].isin([-6, -5, -4, -3, -2, -1]), 0, df['epay01attr21'])
#   df['UnsecuredInstallmentLoanBalance'] = np.where(df['epay01attr19'].isin([-6, -5, -4, -3, -2, -1]), 0, df['epay01attr19'])
#   df['UnsecuredInstallmentLoanTrades'] = np.where(df['epay01attr22'].isin([-6, -5, -4, -3, -2, -1]), 0, df['epay01attr22'])
#   df['UnsecuredInstallmentTradesOpenedLast12mo'] = np.where(df['epay01attr23'].isin([-6, -5, -4, -3, -2, -1]), 0, df['epay01attr23'])
  
  model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type='double')

  return df.withColumn(output_col, model(struct(*map(col, CM_cols))))

# COMMAND ----------

def apply_tier_CPv6(df, cm_score_var="CM_Credit_Model_Score"):
  df = (df
            .withColumn("PricingTier_v6", when(col(cm_score_var) < 0.047079, 1)
                        .when(col(cm_score_var) < 0.1424,   2)
                        .when(col(cm_score_var) < 0.251362, 3)
                        .when(col(cm_score_var) < 0.330307, 4)
                        .when(col(cm_score_var) < 0.367165, 5)
                        .otherwise(6))
           )
  
  return df

# COMMAND ----------

# def score_CM(df, preprocess=True, cm_features_path = cm_features_path, model_uri="models:/CM_v6/1"):
  
#   if preprocess:
#     df = treat_attributes_credit_model_CPv6(df)

#   features = list(pd.read_csv(cm_features_path, header=None)[0])
#   #features_clean = ['_'.join(col.split('_',2)[:2]) for col in features]
#   print(features)

#   model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
#   df = df.withColumn("CM_Credit_Model_Score", model(struct(*map(col, features))))
  
#   #df = apply_tier_CPv6(df)
#   #output_sub = output.select("PPPCL_CONSUMER_ID", "Model_v6_Score", "PricingTier_v6")
#   #output = df.join(output_sub, df.PPPCL_CONSUMER_ID ==  output_sub.PPPCL_CONSUMER_ID,"inner").drop(output_sub.PPPCL_CONSUMER_ID) # COULD BE TIME CONSUMING
#   return df
  

# def treat_attributes_credit_model_CPv6(df):
  
#   # Preprocess CM
#   df = (df
#             .withColumn("V71_AT104S_pct_all_trd_opned_pst_24mo_all_trd",                      F.col("V71_AT104S"))
#             .withColumn("V71_AT31S_pct_opn_trd_gt_75pct_credln_verif_pst_12mo",               F.col("V71_AT31S"))
#             .withColumn("V71_AT34B_util_for_opn_trd_verif_pst_12mo_excl_mtg_home_equity",     F.col("V71_AT34B"))
#             .withColumn("V71_AT36S_mo_since_most_recent_dq",                                  F.col("V71_AT36S"))
#             .withColumn("V71_BC09S_num_cc_trd_opned_pst_24mo",                                F.col("V71_BC09S"))
#             .withColumn("V71_BC102S_avg_credln_opn_cc_trd_verif_pst_12mo",                    F.col("V71_BC102S"))
#             .withColumn("V71_BC104S_avg_the_opn_cc_trd_utils_verif_pst_12mo",                 F.col("V71_BC104S"))
#             .withColumn("V71_BC20S_mo_since_oldest_cc_trd_opned",                             F.col("V71_BC20S"))
#             .withColumn("V71_BC21S_mo_since_most_recent_cc_trd_opned",                        F.col("V71_BC21S"))
#             .withColumn("V71_BC31S_pct_opn_cc_trd_gt_75pct_credln_verif_pst_12mo",            F.col("V71_BC31S"))
#             .withColumn("V71_BC97A_total_opn_buy_opn_cc_verif_pst_3mo",                       F.col("V71_BC97A"))
#             .withColumn("V71_BC98A_total_opn_buy_opn_cc_verif_pst_12mo",                      F.col("V71_BC98A"))
#             .withColumn("V71_BR20S_mo_since_oldest_bank_rvl_trd_opned",                       F.col("V71_BR20S"))
#             .withColumn("V71_BR31S_pct_opn_bank_rvl_trd_gt_75pct_credln_verif_pst_12mo",      F.col("V71_BR31S"))
#             .withColumn("V71_FI34S_util_for_opn_fin_instlmnt_trd_verif_pst_12mo",             F.col("V71_FI34S"))
#             .withColumn("V71_G001B_num_30_or_more_dpd_ratings_pst_12mo",                      F.col("V71_G001B"))
#             .withColumn("V71_G001S_num_30dpd_ratings_pst_12mo",                               F.col("V71_G001S"))
#             .withColumn("V71_G201A_total_opn_buy_opn_trd_verif_pst_3mo_excl_instlmnt_mtg",    F.col("V71_G201A"))
#             .withColumn("V71_G202A_total_opn_buy_opn_trd_verif_pst_12mo_excl_instlmnt_mtg",   F.col("V71_G202A"))
#             .withColumn("V71_G242F_num_fin_inq_includes_dup_pst_3mo",                         F.col("V71_G242F"))
#             .withColumn("V71_G242S_num_inq_includes_dup_pst_3mo",                             F.col("V71_G242S"))
#             .withColumn("V71_G243F_num_fin_inq_includes_dup_pst_6mo",                         F.col("V71_G243F"))
#             .withColumn("V71_G244F_num_fin_inq_includes_dup_pst_12mo",                        F.col("V71_G244F"))
#             .withColumn("V71_G250B_num_30dpd_or_worse_itm_pst_12mo_excl_med_collect_itm",     F.col("V71_G250B"))
#             .withColumn("V71_G250C_num_30dpd_or_worse_itm_pst_24mo_excl_med_collect_itm",     F.col("V71_G250C"))
#             .withColumn("V71_G960S_num_dedup_inq",                                            F.col("V71_G960S"))
#             .withColumn("V71_G980S_num_dedup_inq_pst_6mo",                                    F.col("V71_G980S"))
#             .withColumn("V71_G990S_num_dedup_inq_pst_12mo",                                   F.col("V71_G990S"))
#             .withColumn("V71_IN36S_mo_since_most_recent_instlmnt_dq",                         F.col("V71_IN36S"))
#             .withColumn("V71_RE102S_avg_credln_opn_rvl_trd_verif_pst_12mo",                   F.col("V71_RE102S"))  ##cmv6
#             .withColumn("V71_RE31S_pct_opn_rvl_trd_gt_75pct_credln_verif_pst_12mo",           F.col("V71_RE31S"))
#             .withColumn("V71_S004S_avg_num_mo_trd_have_been_on_file",                         F.col("V71_S004S")) 
#             .withColumn("V71_S114S_num_dedup_inq_pst_6mo_excl_auto_mtg_inq",                  F.col("V71_S114S")) # reqd for eligibility
#             .withColumn("V71_S204S_total_bal_third_party_collect_verif_pst_12mo",             F.col("V71_S204S"))
#             .withColumn("W55_AGG909_mo_since_max_agg_bnkcrd_bal_over_last_12mo",              F.col("W55_AGG909"))
#             .withColumn("W55_AGG910_max_agg_bnkcrd_util_over_last_3mo",                       F.col("W55_AGG910"))
#             .withColumn("W55_AGGS904_peak_mo_bnkcrd_spend_over_pst_12mo",                     F.col("W55_AGGS904"))
#             .withColumn("W55_BALMAG01_non_mtg_bal_magnitude",                                 F.col("W55_BALMAG01"))
#             .withColumn("W55_BALMAG02_rvl_bal_magnitude",                                     F.col("W55_BALMAG02"))
#             .withColumn("W55_INDEX01_annual_yoy_spend_index",                                 F.col("W55_INDEX01"))
#             .withColumn("W55_INDEX02_most_recent_quarter_yoy_spend_index",                    F.col("W55_INDEX02"))
#             .withColumn("W55_PAYMNT08_ratio_actual_min_pmt_for_rvl_trd_last_mo",              F.col("W55_PAYMNT08"))
#             .withColumn("W55_PAYMNT10_num_pmt_last_3mo",                                      F.col("W55_PAYMNT10"))
#             .withColumn("W55_PAYMNT11_num_pmt_last_12mo",                                     F.col("W55_PAYMNT11"))
#             .withColumn("W55_REVS904_max_agg_rvl_mo_spend_over_last_12mo",                    F.col("W55_REVS904"))
#             .withColumn("W55_RVDEX01_annual_yoy_rvl_spend_index",                             F.col("W55_RVDEX01"))
#             .withColumn("W55_RVDEX02_most_recent_quarter_yoy_rvl_spend_index",                F.col("W55_RVDEX02"))
#             .withColumn("W55_RVLR01_util_for_bnkcrd_acct_with_a_rvl_bal",                     F.col("W55_RVLR01"))
#             .withColumn("W55_TRV01_num_mo_since_overlimit_on_a_bnkcrd",                       F.col("W55_TRV01"))
#             .withColumn("W55_TRV02_num_mo_overlimit_on_a_bnkcrd_over_last_12mo",              F.col("W55_TRV02"))
#             .withColumn("W55_TRV03_num_non_mtg_trd_with_a_bal_incr_last_mo",                  F.col("W55_TRV03"))
#             .withColumn("W55_TRV04_num_non_mtg_bal_incr_last_3mo",                            F.col("W55_TRV04"))
#             .withColumn("W55_TRV07_num_non_mtg_bal_decr_last_mo",                             F.col("W55_TRV07"))
#             .withColumn("W55_TRV08_num_non_mtg_bal_decr_last_3mo",                            F.col("W55_TRV08"))
#             .withColumn("W55_TRV09_num_non_mtg_bal_decr_yoy",                                 F.col("W55_TRV09"))
#             .withColumn("W55_TRV10_num_mo_non_mtg_bal_decr_last_12mo",                        F.col("W55_TRV10"))
#             .withColumn("W55_TRV11_num_rvl_high_cred_incr_last_mo",                           F.col("W55_TRV11"))
#             .withColumn("W55_TRV12_num_rvl_high_cred_incr_last_3mo",                          F.col("W55_TRV12"))
#             .withColumn("W55_TRV13_num_rvl_high_cred_incr_yoy",                               F.col("W55_TRV13"))
#             .withColumn("W55_TRV22_num_mo_bnkcrd_cred_limit_decr_last_12mo",                  F.col("W55_TRV22"))
#             .withColumn("V71_AT06S_num_trd_opned_pst_6mo",                                    F.col("V71_AT06S"))
#             .withColumn("V71_AT09S_num_trd_opned_pst_24mo",                                   F.col("V71_AT09S"))
#             .withColumn("V71_AT21S_mo_since_most_recent_trd_opned",                           F.col("V71_AT21S"))
#             .withColumn("V71_AT28A_total_credln_opn_trd_verif_pst_12mo",                      F.col("V71_AT28A"))
#             .withColumn("V71_AT30S_pct_opn_trd_more_than_50pct_credln_verif_pst_12mo",        F.col("V71_AT30S"))
#             .withColumn("V71_AT32S_max_bal_owed_on_opn_trd_verif_pst_12mo",                   F.col("V71_AT32S"))
#             .withColumn("V71_AU36S_mo_since_most_recent_auto_dq",                             F.col("V71_AU36S"))       
#             .withColumn("V71_BC34S_util_for_opn_cc_trd_verif_pst_12mo",                       F.col("V71_BC34S"))
#             .withColumn("V71_BC35S_avg_bal_opn_cc_trd_verif_pst_12mo",                        F.col("V71_BC35S"))
#             .withColumn("V71_BR09S_num_bank_rvl_trd_opned_pst_24mo",                          F.col("V71_BR09S"))
#             .withColumn("V71_BR21S_mo_since_most_recent_bank_rvl_trd_opned",                  F.col("V71_BR21S"))
#             .withColumn("V71_FI06S_num_fin_instlmnt_trd_opned_pst_6mo",                       F.col("V71_FI06S"))
#             .withColumn("V71_FI09S_num_fin_instlmnt_trd_opned_pst_24mo",                      F.col("V71_FI09S"))
#             .withColumn("V71_FI30S_pct_opn_fin_instlmnt_trd_gt_50pct_credln_verif_pst_12mo",  F.col("V71_FI30S"))
#             .withColumn("V71_FI31S_pct_opn_fin_instlmnt_trd_gt_75pct_credln_verif_pst_12mo",  F.col("V71_FI31S"))
#             .withColumn("V71_G058S_num_trd_30_or_more_dpd_pst_6mo",                           F.col("V71_G058S"))
#             .withColumn("V71_G059S_num_trd_30_or_more_dpd_pst_12mo",                          F.col("V71_G059S"))
#             .withColumn("V71_G061S_num_trd_30_or_more_dpd_pst_24mo",                          F.col("V71_G061S"))
#             .withColumn("V71_G213A_highest_bal_third_party_collect_verif_24mo",               F.col("V71_G213A"))
#             .withColumn("V71_G213B_highest_bal_non_med_third_party_collect_verif_24mo",       F.col("V71_G213B"))
#             .withColumn("V71_G215A_num_third_party_collect_with_bal_larger_than_0_dollar",    F.col("V71_G215A"))
#             .withColumn("V71_G234S_num_day_with_inquiry_occurring_pst_30day",                 F.col("V71_G234S"))
#             .withColumn("V71_G237S_num_inq_pst_6mo_includes_dup",                             F.col("V71_G237S"))
#             .withColumn("V71_G238S_num_inq_pst_12mo_includes_dup",                            F.col("V71_G238S"))
#             .withColumn("V71_G244S_num_inq_pst_12mo_includes_dup",                            F.col("V71_G244S"))
#             .withColumn("V71_G251A_num_60dpd_or_worse_itm_ever_excl_med_collect_itm",         F.col("V71_G251A"))
#             .withColumn("V71_G251B_num_60dpd_or_worse_itm_pst_12mo_excl_med_collect_itm",     F.col("V71_G251B"))
#             .withColumn("V71_G310S_worst_rating_on_all_trd_pst_12mo",                         F.col("V71_G310S"))
#             .withColumn("V71_MT20S_mo_since_oldest_mtg_trd_opned",                            F.col("V71_MT20S"))
#             .withColumn("V71_MT28S_total_credln_opn_mtg_trd_verif_pst_12mo",                  F.col("V71_MT28S"))
#             .withColumn("V71_MT36S_mo_since_most_recent_mtg_dq",                              F.col("V71_MT36S"))
#             .withColumn("V71_PB20S_mo_since_oldest_premium_cc_trd_opned",                     F.col("V71_PB20S"))
#             .withColumn("V71_RE09S_num_rvl_trd_opned_pst_24mo",                               F.col("V71_RE09S"))
#             .withColumn("V71_RE20S_mo_since_oldest_rvl_trd_opned",                            F.col("V71_RE20S"))
#             .withColumn("V71_RE28S_total_credln_opn_rvl_trd_verif_pst_12mo",                  F.col("V71_RE28S"))
#             .withColumn("V71_RE30S_pct_opn_rvl_trd_gt_50pct_credln_verif_pst_12mo",           F.col("V71_RE30S"))
#             .withColumn("V71_RE34S_util_for_opn_rvl_trd_verif_pst_12mo",                      F.col("V71_RE34S"))
#             .withColumn("V71_RT31S_pct_opn_retail_trd_gt_75pct_credln_verif_pst_12mo",        F.col("V71_RT31S"))
#             .withColumn("V71_S043S_num_opn_trd_gt_50pct_credln_verif_pst_12mo_excl_instlmnt_mtg",F.col("V71_S043S"))
#             .withColumn("V71_ST36S_mo_since_most_recent_student_ln_dq",                       F.col("V71_ST36S"))
#             .withColumn("V71_ST99S_total_bal_all_student_ln_trd_ever_dq",                     F.col("V71_ST99S"))
#             .withColumn("W55_AGG904_num_agg_non_mtg_cred_limit_decr_over_last_quarter",       F.col("W55_AGG904"))
#             .withColumn("W55_AGG911_max_agg_bnkcrd_util_over_last_year",                      F.col("W55_AGG911"))
#             .withColumn("W55_REVS901_agg_rvl_mo_spend_over_last_3mo",                         F.col("W55_REVS901"))
#             .withColumn("W55_TRV17_num_bnkcrd_acct_with_a_yoy_cred_limit_incr",               F.col("W55_TRV17"))
#             .withColumn("W55_TRV21_num_bnkcrd_acct_with_a_yoy_cred_limit_decr",               F.col("W55_TRV21"))
#             .withColumn("V71_AT20S_mo_since_oldest_trd_opn",                                  F.col("V71_AT20S"))) # reqd for eligibility
  
#   df = (df
#             .withColumn("W55_TRV07_num_non_mtg_bal_decr_last_mo", 
#                         F.when((col("W55_TRV07_num_non_mtg_bal_decr_last_mo").isin([-6, -5, -4, -3, -2, -1])), 999).otherwise(F.col("W55_TRV07_num_non_mtg_bal_decr_last_mo")))
#             .withColumn("W55_TRV08_num_non_mtg_bal_decr_last_3mo", 
#                         F.when((col("W55_TRV08_num_non_mtg_bal_decr_last_3mo").isin([-6, -5, -4, -3, -2, -1])), 999).otherwise(F.col("W55_TRV08_num_non_mtg_bal_decr_last_3mo")))
#             .withColumn("W55_TRV09_num_non_mtg_bal_decr_yoy", 
#                         F.when((col("W55_TRV09_num_non_mtg_bal_decr_yoy").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("W55_TRV09_num_non_mtg_bal_decr_yoy")))
#             .withColumn("W55_TRV10_num_mo_non_mtg_bal_decr_last_12mo", 
#                         F.when((col("W55_TRV10_num_mo_non_mtg_bal_decr_last_12mo").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("W55_TRV10_num_mo_non_mtg_bal_decr_last_12mo")))
            
#             .withColumn("W55_AGG909_mo_since_max_agg_bnkcrd_bal_over_last_12mo", 
#                         F.when((col("W55_AGG909_mo_since_max_agg_bnkcrd_bal_over_last_12mo").isin([-6,-5,-4,-3,-2,-1])), 999).otherwise( F.col("W55_AGG909_mo_since_max_agg_bnkcrd_bal_over_last_12mo")))
#             .withColumn("W55_TRV17_num_bnkcrd_acct_with_a_yoy_cred_limit_incr", 
#                         F.when((col("W55_TRV17_num_bnkcrd_acct_with_a_yoy_cred_limit_incr").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("W55_TRV17_num_bnkcrd_acct_with_a_yoy_cred_limit_incr")))
            
#             .withColumn("V71_AT36S_mo_since_most_recent_dq", 
#                         F.when((col("V71_AT36S_mo_since_most_recent_dq").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("V71_AT36S_mo_since_most_recent_dq")))
#             .withColumn("V71_AU36S_mo_since_most_recent_auto_dq", 
#                         F.when((col("V71_AU36S_mo_since_most_recent_auto_dq").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("V71_AU36S_mo_since_most_recent_auto_dq")))
#             .withColumn("V71_IN36S_mo_since_most_recent_instlmnt_dq", 
#                         F.when((col("V71_IN36S_mo_since_most_recent_instlmnt_dq").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("V71_IN36S_mo_since_most_recent_instlmnt_dq")))
#             .withColumn("V71_MT36S_mo_since_most_recent_mtg_dq", 
#                         F.when((col("V71_MT36S_mo_since_most_recent_mtg_dq").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("V71_MT36S_mo_since_most_recent_mtg_dq")))
#             .withColumn("V71_ST36S_mo_since_most_recent_student_ln_dq", 
#                         F.when((col("V71_ST36S_mo_since_most_recent_student_ln_dq").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("V71_ST36S_mo_since_most_recent_student_ln_dq")))
#             .withColumn("W55_TRV01_num_mo_since_overlimit_on_a_bnkcrd", 
#                         F.when((col("W55_TRV01_num_mo_since_overlimit_on_a_bnkcrd").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("W55_TRV01_num_mo_since_overlimit_on_a_bnkcrd")))
            
#             .withColumn("W55_PAYMNT08_ratio_actual_min_pmt_for_rvl_trd_last_mo", 
#                         F.when((col("W55_PAYMNT08_ratio_actual_min_pmt_for_rvl_trd_last_mo").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("W55_PAYMNT08_ratio_actual_min_pmt_for_rvl_trd_last_mo")))
#             .withColumn("V71_AT21S_mo_since_most_recent_trd_opned", 
#                         F.when((col("V71_AT21S_mo_since_most_recent_trd_opned").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("V71_AT21S_mo_since_most_recent_trd_opned")))
#             .withColumn("V71_BC21S_mo_since_most_recent_cc_trd_opned", 
#                         F.when((col("V71_BC21S_mo_since_most_recent_cc_trd_opned").isin([-6, -5, -4, -3, -2, -1])), 999 ).otherwise(F.col("V71_BC21S_mo_since_most_recent_cc_trd_opned")))
#             .withColumn("V71_BR21S_mo_since_most_recent_bank_rvl_trd_opned", F.when((col("V71_BR21S_mo_since_most_recent_bank_rvl_trd_opned").isin(-6, -5, -4, -3, -2, -1)), 999 ).otherwise(F.col("V71_BR21S_mo_since_most_recent_bank_rvl_trd_opned")))
            
#             .withColumn("DaysSinceMostRecentUILInquiry",            F.when((col("epay01attr26").isin([-6, -5, -4, -3, -2, -1])), 1e+05).otherwise(col("epay01attr26")))
#             .withColumn("MaxUtilizationOfUnsecuredInstallmentLoan", F.when((col("epay01attr21").isin([-6, -5, -4, -3, -2, -1])), 0).otherwise(col("epay01attr21")))
#             .withColumn("UnsecuredInstallmentLoanBalance",          F.when((col("epay01attr19").isin([-6, -5, -4, -3, -2, -1])), 0).otherwise(col("epay01attr19")))
#             .withColumn("UnsecuredInstallmentLoanTrades",           F.when((col("epay01attr22").isin([-6, -5, -4, -3, -2, -1])), 0).otherwise(col("epay01attr22")))
#             .withColumn("UnsecuredInstallmentTradesOpenedLast12mo", F.when((col("epay01attr23").isin([-6, -5, -4, -3, -2, -1])), 0).otherwise(col("epay01attr23")))
            
#     )
  
#   return df
# ###########################################################################################################################
# def apply_tier_CPv6(df, cm_score_var="CM_Credit_Model_Score"):
#   output = (df
#             .withColumn("PricingTier_v6", when(col(cm_score_var) < 0.047079, 1)
#                         .when(col(cm_score_var) < 0.1424,   2)
#                         .when(col(cm_score_var) < 0.251362, 3)
#                         .when(col(cm_score_var) < 0.330307, 4)
#                         .when(col(cm_score_var) < 0.367165, 5)
#                         .otherwise(6))
#            )
#   return output
