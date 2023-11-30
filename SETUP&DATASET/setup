import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('rainbow')
sns.set_style('whitegrid')
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
sns.set_style('darkgrid')

# pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.sql.types import StructType
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col, count, desc, max
from pyspark.ml.feature import  StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
CSV_FILE_1='Final_database.csv'
CSV_FILE_2= 'Database_to_calculate_popularity.csv'
get_ipython().system('pip install --upgrade pandas')
spark = SparkSession.builder.master("local[]").appName("Spotify-Song-Recommender").getOrCreate()
spark
sc = spark.sparkContext
sqlContext = SQLContext(sc)

df = spark.read.option("header", True).csv(CSV_FILE_1)
df = df.withColumn('Release_date', F.to_date('Release_date', "yyyy-MM-dd"))
numerical_features = ['danceability', 'energy', 'instrumentalness', 'valence', 'liveliness', 'speechiness', 'acoustics',
                      'instrumentalness', 'tempo', 'duration_ms','time_signature', 'Days_since_release', 'n_words']

for c in numerical_features:
    df = df.withColumn(c, df[c].cast("float"))
    
cols_to_drop = ['syuzhet_norm', 'bing_norm', 'afinn_norm', 'nrc_norm', 'syuzhet', 'bing'] 
for c in cols_to_drop:
    df.drop(c).collect()
    
df.printSchema()

# Use os.path to join paths
CSV_FILE_2= 'Database_to_calculate_popularity.csv'

# Read the CSV file into a DataFrame
df_listenings = spark.read.format('csv').option('header', True).option('inferSchema', True).load(CSV_FILE_2)

# Show the DataFrame
df_listenings.show()

df_listenings = df_listenings.drop('date','country','uri') #drops date column
df_listenings = df_listenings.na.drop() # removes null values in the row
df_listenings.show()

rows = df_listenings.count()
cols = len(df_listenings.columns)
print(rows,cols)
