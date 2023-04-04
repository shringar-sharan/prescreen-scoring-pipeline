# Databricks notebook source
# DBTITLE 1,Read Databricks dataset  
# Load the data from its source.
df = spark.read.load('/databricks-datasets/learning-spark-v2/people/people-10m.delta')

# Show the results.
display(df)

# COMMAND ----------

# DBTITLE 1,Write out DataFrame as Databricks Delta data
table_name = "people_10m"

# Write the data to a table.
df.write.saveAsTable(table_name)

# COMMAND ----------

# DBTITLE 1,Query the table
# Load the data from the save location.
people_df = spark.read.table(table_name)

display(people_df)

# COMMAND ----------

# DBTITLE 1,Visualize data
display(people_df.select('gender').orderBy('gender', ascending = False).groupBy('gender').count())

# COMMAND ----------

display(people_df.select("salary").orderBy("salary", ascending = False))

# COMMAND ----------

# DBTITLE 1,Optimize table 
display(spark.sql(f"OPTIMIZE {table_name}"))

# COMMAND ----------

# DBTITLE 1,Show table history
display(spark.sql(f"DESCRIBE HISTORY {table_name}"))

# COMMAND ----------

# DBTITLE 1,Show table details
display(spark.sql(f"DESCRIBE DETAIL {table_name}"))

# COMMAND ----------

# DBTITLE 1,Show the table format
display(spark.sql(f"DESCRIBE FORMATTED {table_name}"))

# COMMAND ----------

# DBTITLE 1,Clean up
# Delete the table.
spark.sql("DROP TABLE {table_name}")
