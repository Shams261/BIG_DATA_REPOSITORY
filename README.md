# Spotify Song Data Analysis with PySpark

## Overview
This project utilizes PySpark, a Python library for Apache Spark, to analyze Spotify song data. The script covers a range of tasks, including data exploration, clustering, dimensionality reduction, and collaborative filtering for song recommendations.

## Dataset
The analysis is conducted on two main datasets:
1. Final Spotify Database (Final_database.csv):
This dataset contains detailed information about various Spotify songs, including features like danceability, energy, instrumentalness, valence, and more.
2. Database for Calculating Popularity (Database_to_calculate_popularity.csv):
This dataset includes information about the popularity and listening statistics of songs, such as position, track URI, and country.

Both datasets are utilized to extract insights and patterns related to song popularity, artist trends, and user preferences.

## Requirements
Python 

PySpark

Pandas

Seaborn

Scikit-learn

Sql

Hdfs

Clustering

Mapreduce

## Install the required dependencies:

pip install pyspark pandas seaborn scikit-learn

## EDA(Exploratory Data Analysis)

EDA is the procedure which is used to gather deep and hidden information about the dataset by categorizing the data in various different ways such as finding duplictes, finding and handling null values,and visulaization of data through plots, charts and graphs.
Here we are performing SQL queries, Plots and Figures, and using pyspark to filter results from the dataset.

![image](https://github.com/Shams261/BIG_DATA_REPOSITORY/assets/56577910/7dccfd36-9404-4c4c-bf41-2243b6d03789)

## Clustering
Clustering is the task of dividing the unlabeled data or data points into different clusters such that similar data points fall in the same cluster than those which differ from the others. In simple words, the aim of the clustering process is to segregate groups with similar traits and assign them into clusters.

![image](https://github.com/Shams261/BIG_DATA_REPOSITORY/assets/56577910/83dd59d9-6945-48bf-8de7-170743d59b59)

## Key Features

1. Data Loading:The script loads Spotify song data from CSV files (Final_database.csv and Database_to_calculate_popularity.csv) using PySpark.











