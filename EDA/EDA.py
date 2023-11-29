# EDA(Exploratory Data Analysis)
# EDA is the procedure which is used to gather deep and hidden information about the dataset by categorizing the data in various different ways such as finding duplictes, finding and handling null values,and visulaization of data through plots, charts and graphs.
# Here we are performing SQL queries, Plots and Figures, and using pyspark to filter results from the dataset.

# Getting Unique songs in the dataset
df.select(["Title","Artist"]).distinct().count()
print("Dataset Shape using spark syntax:\n",(df.count(), len(df.columns)))

# Top 200 most played songs on the spotify in past three years

result_df = (df.groupBy("Artist")
               .count()
               .orderBy("count", ascending=False)
               .limit(10)
               .toPandas()
            )

# Plotting the barplot of output songs
custom_colors = ["red", "green", "blue", "orange", "purple", "pink", "brown", "gray", "cyan", "magenta"]
sns.barplot(data=result_df, y='Artist', x='count', palette=custom_colors)
plt.title('Most Prolific Artists')
plt.show()


df.createOrReplaceTempView("df_table")

# Cross checking the dataset length
print("Now using the SQL Context. We can check it's the same length as before")
query = """
    SELECT Count(*) as Dataset_Length
    FROM df_table
"""
res = spark.sql(query).show()

# Finding the most popular artist by doing addition of their popular songs (USA)
query = """
SELECT
       Artist, 
       ROUND(SUM(Popularity), 2) AS Populartiy
FROM df_table
WHERE USA == 1
GROUP BY Artist
ORDER BY AVG(Popularity) DESC
LIMIT 10
"""

res = spark.sql(query)
res.show(10, truncate=False)

# Showing the top 10 songs released in 1970
(df.filter(F.year(df['Release_date']) == 1970)
   .select('Title', 'Artist','Release_date', 'Genre')
   .distinct()
   .show(10, truncate=False)
)


#Most Popular Song per Decade

# Getting the most popular song per decade
query = """
SELECT
    ROUND(Year(Release_date), -1) as Decade,
    ROUND(Max(Popularity), 2) as Popularity,
    SUBSTRING(MAX(CONCAT(LPAD(Popularity, 11, 0), Title)), 12) AS Title,
    SUBSTRING(MAX(CONCAT(LPAD(Popularity, 11, 0), Artist)), 12) AS Artist
FROM
    df_table
WHERE
    ROUND(Year(Release_date), -1) IS NOT NULL
    AND USA == 1
GROUP BY Decade
ORDER BY Decade ASC
"""

spark.sql(query).show()

#Most popular Genre per decade

#Most popular genres, period.

query = """
SELECT Genre, COUNT(*) AS Tally
FROM df_table
GROUP BY Genre
ORDER BY Tally DESC
"""
spark.sql(query).show(5)

# Printing the genre and decade 
query = """
SELECT
      ROUND(Year(Release_date), -1) AS Decade,
      Genre, COUNT(Genre) AS counts
FROM  df_table
WHERE ROUND(Year(Release_date), -1) IS NOT NULL
GROUP BY Decade, Genre
ORDER BY COUNT(Genre) DESC
"""

res = (spark.sql(query)
            .dropDuplicates(subset=['Decade'])
            .orderBy('Decade')
            .show()
      )


# ## Most popular day for each track!
# (checking for few data)

query = """
SELECT Title, Artist, Release_date, MAX(Popularity)
FROM df_table
WHERE Artist == "Paulo Londra"
GROUP BY Title, Artist, Release_date
LIMIT 10
"""

res = spark.sql(query).show()


# ## Changing of music patterns over the decade 

sound_features = ['danceability', 'energy', 'instrumentalness', 'valence', 'liveliness', 'speechiness', 'acoustics']
col_names = ['Decade']
col_names.extend(sound_features)

df_music_features = (df.sample(.2, seed=42)
                       .groupBy(F.round(F.year(df.Release_date), -1))
                       .agg({feature: 'mean' for feature in sound_features})
                       .toDF(*col_names)
                       .orderBy('Decade')
                       .toPandas()
                       .dropna(axis=0)
                    )
sns.lineplot(data=pd.melt(df_music_features, ['Decade']), x='Decade', y='value', hue='variable').set_title('Characteristics of song over the Decades');
