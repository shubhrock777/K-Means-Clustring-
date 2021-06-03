import pandas as pd 

crime_df = pd.read_csv("D:/BLR10AM/Assi/06Hierarchical clustering/Dataset_Assignment Clustering/crime_data.csv")

#details of dataframe
crime_df.describe()
crime_df.info()

#checking for null or na vales 
crime_df.isna().sum()
crime_df.isnull().sum()

col_uni = crime_df.nunique()


########exploratory data analysis
EDA = {"columns_name ":crime_df.columns,
                  "mean":crime_df.mean(),
                  "median":crime_df.median(),
                  "standard_deviation":crime_df.std(),
                  "variance":crime_df.var(),
                  "skewness":crime_df.skew(),
                  "kurtosis":crime_df.kurt()}

EDA


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(crime_df.iloc[:,1 :])

import matplotlib.pyplot as plt

#boxplot for every column
crime_df.columns
boxplot = crime_df.boxplot(column=[ 'Murder', 'Assault', 'UrbanPop', 'Rape'])

#column num 1 has no name 
crime_df = crime_df.rename(columns = {"Unnamed: 0":"state"})

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime_df.iloc[:, 1:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering cheacking 4/5/6 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering



########## with 4 cluster 
with_4clust = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
with_4clust.labels_

cluster4_labels = pd.Series(with_4clust.labels_)

crime_df['clust4'] = cluster4_labels # creating a new column and assigning it to new column 


# Aggregate mean of each cluster
clust4_details  = crime_df.iloc[:, 1:5].groupby(crime_df.clust4).mean()
clust4_details

########## with 5 cluster 
with_5clust = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
with_5clust.labels_

cluster5_labels = pd.Series(with_5clust.labels_)

crime_df['clust5'] = cluster5_labels # creating a new column and assigning it to new column 


# Aggregate mean of each cluster
clust5_details  = crime_df.iloc[:, 1:5].groupby(crime_df.clust5).mean()
clust5_details  


########## with 6 cluster 
with_6clust = AgglomerativeClustering(n_clusters = 6, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
with_6clust.labels_

cluster6_labels = pd.Series(with_6clust.labels_)

crime_df['clust6'] = cluster6_labels # creating a new column and assigning it to new column 


# Aggregate mean of each cluster
clust6_details  = crime_df.iloc[:, 1:5].groupby(crime_df.clust6).mean()
clust6_details



###########final data with 4 clusters 

crime_final = crime_df.iloc[:, [5,0,1,2,3,4]]
crime_final.head()

# Aggregate mean of each cluster
fclust_details  = crime_final.iloc[:, 1:].groupby(crime_final.clust4).mean()
fclust_details


# creating a csv file  for new data frame with cluster 
crime_final.to_csv("crime_final.csv", encoding = "utf-8")

# creating a csv file  with details of each cluster 
fclust_details.to_csv("fclust_details.csv", encoding = "utf-8")