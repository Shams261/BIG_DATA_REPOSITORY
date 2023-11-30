#!/usr/bin/env python
# coding: utf-8
# Clustering
# Clustering is the task of dividing the unlabeled data or data points into different clusters such that similar data points fall in the same cluster than those which differ from the others. In simple words, the aim of the clustering process is to segregate groups with similar traits and assign them into clusters.
# There are several techniques of comparing music similarities. One option is to simply use continuous, numerical variables (such as danceability, energy, and so on) and reduce dimensionality using PCA, k-means, or some other method and if we are only interested in the song features (continuous variables), we could simply generate a feature vector and use cosine similarity to discover the most similar sounding song while accounting for numerical attributes and one-hot-encoded countries. 
# Checking the cosine similarity of song feature vectors
# for pop
query_kpop = """
SELECT Title, Artist, {}
FROM df_table
WHERE `k-pop` = 1
""".format(', '.join(numerical_features))

df_kpop_songs = (spark.sql(query_kpop)
                      .sample(.1)
                      .dropna()
                      .toPandas()
                )

# for rap
query_rap = """
SELECT Title, Artist, {}
FROM df_table
WHERE rap = 1
""".format(', '.join(numerical_features))

df_rap_songs = (spark.sql(query_rap)
                     .sample(.1)
                     .dropna()
                     .toPandas()
               )
df_rap_songs.head()
df_kpop_songs.head()
from scipy import spatial
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
transformer = Normalizer()

# drop the title and artist with the iloc
scaled_kpop_df = scaler.fit_transform(df_kpop_songs.iloc[:, 2:]) 
scaled_rap_df = scaler.fit_transform(df_rap_songs.iloc[:, 2:])

# printing the cosine similarity of a rap and a k-pop song
song1 = np.array(scaled_rap_df[1])
song2 = np.array(scaled_kpop_df[2])
result = 1 - spatial.distance.cosine(song1, song2)
print("Cosine similarity of a rap and a k-pop song:", result)

# Printing the cosine similarity of two rap songs
song1 = np.array(scaled_rap_df[1])
song2 = np.array(scaled_rap_df[10])
result = 1 - spatial.distance.cosine(song1, song2)
print("Cosine similarity of two rap songs:", result)

# Dimensionality Reduction
# Dimensionality reduction refers to techniques for reducing the number of input variables in training data.This is useful for visualizing kmeans clustering later.In this case, we're going to take two types of music (Kpop and Rap), and then try reducing all the numeric, musical features down to two dimentions. The two dimentions won't really represent the genre of the music, but we can pretend that this is true. When we do KMeans clustering later on, we can visualize it on these two PCA axis.
df_rap_songs = df_rap_songs.assign(is_rap=1,
                                   is_kpop=0
                                   )
df_kpop_songs = df_kpop_songs.assign(is_rap=0,
                                     is_kpop=1
                                     )
df_rap_and_kpop = pd.concat([df_rap_songs, df_kpop_songs])
X = scaler.fit_transform(df_rap_and_kpop.iloc[:, 2:])
pca = PCA(n_components=10)
pca.fit(X)
print(pca.explained_variance_ratio_)
sns.lineplot(x=[x for x in range(1, 11)], y=pca.explained_variance_ratio_).set_title("% Variance Explained vs # Dimensions");

# As expected, it's able to explain most the vairance using 1 dimension. This roughly corresponds to "genre," which instead was encoded as either `is_rap` or `is_kpop`
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
PCA_components = pd.DataFrame(principalComponents)
# sns.scatterplot(data=principalComponents, alpha=.1)
sns.scatterplot(x=PCA_components[0], y=PCA_components[1], alpha=.1).set_title("First 2 PCA Components");
plt.xlabel('PCA 1');
plt.ylabel('PCA 2');

# The first component is particularly excellent at separation
# Kmeans Clustering
# K-means clustering is a method for grouping n observations into K clusters
# Finding the best number of clusters with an elbow plot. Viewed the top 2 PCA clusters, and then used kmeans with various number of clusters. The "Scree" plot below, shows the percent of variance explained as a function of the number of clusters used
ks = range(1, 10)
inertias = []
for k in ks:
    # Creating a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    # Fitting model to samples
    model.fit(PCA_components.iloc[:,:2])
    # Appending the inertia to the list of inertias
    inertias.append(model.inertia_)    
sns.lineplot(x=ks, y=inertias, marker='o').set_title("Inertia vs # Clusters used")
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# Again, as expected, 2 clusters seems to make sense
km = KMeans(
    n_clusters=2, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)
df_pca_kmeans_plot = pd.concat([PCA_components, pd.Series(y_km)], axis=1)
df_pca_kmeans_plot.columns = ['PCA_1', 'PCA_2', 'Cluster']
sns.scatterplot(data=df_pca_kmeans_plot, x='PCA_1', y='PCA_2', hue='Cluster')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 component PCA');

#Coloring the genre instead of cluster
df_final = pd.concat([df_pca_kmeans_plot, df_rap_and_kpop.reset_index()['is_rap']], axis=1)
df_final['is_rap'] = df_final['is_rap'].replace({1:'Rap', 0: 'KPop'})
df_final['Cluster'] = df_final['Cluster'].replace({1:'Cluster 2', 0: 'Cluster 1'})
df_final = df_final.rename(columns={'is_rap': 'Genre'})
sns.scatterplot(data=df_final, x='PCA_1', y='PCA_2', hue='Genre')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 component PCA');
df_rap_and_kpop = pd.concat([df_rap_songs, df_kpop_songs])
rap_kpop_labels = df_rap_and_kpop[['is_rap', 'is_kpop']]
df_rap_and_kpop = df_rap_and_kpop.drop(columns=['is_rap', 'is_kpop'])
X = scaler.fit_transform(df_rap_and_kpop.iloc[:, 2:])
pca = PCA(n_components=10)
pca.fit(X)
print(pca.explained_variance_ratio_)
sns.lineplot(x=[x for x in range(1, 11)], y=pca.explained_variance_ratio_).set_title("% Variance Explained vs # Dimensions")
plt.show()
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
PCA_components = pd.DataFrame(principalComponents)

# Doing scatterplot where data is equal to principalComponents and alpha=0.1
sns.scatterplot(x=PCA_components[0], y=PCA_components[1], alpha=.1).set_title("First 2 PCA Components");
plt.xlabel('PCA 1');
plt.ylabel('PCA 2');
plt.show()
ks = range(1, 10)
inertias = []
for k in ks:
    # Creating a KMeans instance with k clusters model
    model = KMeans(n_clusters=k)
    # Fitting model to samples
    model.fit(PCA_components.iloc[:,:2])
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
sns.lineplot(x=ks, y=inertias, marker='o').set_title("Inertia vs # Clusters used")
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)
df_pca_kmeans_plot = pd.concat([PCA_components, pd.Series(y_km)], axis=1)
df_pca_kmeans_plot.columns = ['PCA_1', 'PCA_2', 'Cluster']
sns.scatterplot(data=df_pca_kmeans_plot, x='PCA_1', y='PCA_2', hue='Cluster')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 component PCA');
df_final = pd.concat([df_pca_kmeans_plot, rap_kpop_labels.reset_index()['is_rap']], axis=1)
df_final['is_rap'] = df_final['is_rap'].replace({1:'Rap', 0: 'KPop'})
df_final['Cluster'] = df_final['Cluster'].replace({1:'Cluster 2', 0: 'Cluster 1'})
df_final = df_final.rename(columns={'is_rap': 'Genre'})
sns.scatterplot(data=df_final, x='PCA_1', y='PCA_2', hue='Genre')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 component PCA');

