#After performing EDA and Clusturing we are intialising the model to test with the user input

df_listenings_agg = df_listenings.select('position', 'track').groupby('position', 'track').agg(count('*').alias('count')).orderBy('position')

df_listenings_agg.show()

row = df_listenings_agg.count()
col = len(df_listenings_agg.columns)
print(row,col)

df_listenings_agg = df_listenings_agg.limit(50000)

old_strindexer = [StringIndexer(inputCol = col, outputCol = col + '_index').fit(df_listenings_agg) for col in list(set(df_listenings_agg.columns)- set(['count']))]
indexer = [curr_strindexer.setHandleInvalid("keep") for curr_strindexer in old_strindexer]
stages = [(f"indexer_{i}", curr_indexer) for i, curr_indexer in enumerate(indexer)]
pipeline = Pipeline(stages = indexer)
data = pipeline.fit(df_listenings_agg).transform(df_listenings_agg)
data.show()

data = data.select('position_index', 'track_index', 'count').orderBy('position_index')
data.show()

# Splitting the data
(training, test) = data.randomSplit([0.5,0.5])
POSITION = "position_index"
TRACK = "track_index"
COUNT = "count"
als = ALS(maxIter = 5, regParam = 0.01, userCol = POSITION, itemCol = TRACK, ratingCol = COUNT)

# Alternating Least Squares algorithm
model = als.fit(training)
predictions = model.transform(test)

recs = model.recommendForAllUsers(10)
recs.show()

# Showing the 10 recommendations for 2 user
recs.take(2)

# Finding the most similar song
query_all = """
SELECT Title, Artist, Genre, {}
FROM df_table
""".format(', '.join(numerical_features))

df_all_songs = (spark.sql(query_all)
                     .dropna()
                     .toPandas()
                     .drop_duplicates(['Title', 'Artist'])
                     .reset_index(drop=True)
                )

df_all_songs.columns

df_all_songs_ohe = pd.get_dummies(df_all_songs.drop(columns='Title'))
scaled_df_all_songs_ohe = scaler.fit_transform(df_all_songs_ohe)


def get_most_similar_song(title, artist):
    title = title.lower()
    
    # getting the vector for the requested song
    song_idx = df_all_songs.query(f"Title == '{title}' and Artist == '{artist}'").index.values[0]
    song_vector = scaled_df_all_songs_ohe[song_idx]
    
    # finding the most similar song
    min_difference = 1
    closest_song_idx = 0
    for index, song in enumerate(scaled_df_all_songs_ohe):
        distance = spatial.distance.cosine(song_vector, song)
        if distance < min_difference:
            if index == song_idx:
                pass
            else:
                min_difference = distance
                closest_song_idx = index
    
    # getting the title and the artist of the most similar song
    closest_song = df_all_songs.loc[closest_song_idx,['Title', 'Artist']]
    print("Closest Song:\n-------------", closest_song, sep="\n")
    return closest_song_idx

get_most_similar_song("Numb", "Linkin Park")


spark.sql("SELECT Artist, Title FROM df_table WHERE Artist LIKE 'Radio%'").distinct().show(50)


# Showing some recommendations based on the user input

get_most_similar_song("it wont kill ya", "The Chainsmokers - Louane")

get_most_similar_song("how would you feel", "Ed Sheeran")

get_most_similar_song("the box", "Roddy Ricch")
