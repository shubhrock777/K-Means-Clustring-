import pandas as pd

import matplotlib.pyplot as plt

insu = pd.read_csv("D:/BLR10AM/Assi/06Hierarchical clustering/Dataset_Assignment Clustering/AutoInsurance.csv")


#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":insu.columns,
                "data types ":insu.dtypes})


###########Data Pre-processing 

#unique value for each columns 
col_uni =insu.nunique()
col_uni


#details of dataframe
insu.describe()
insu.info()

#checking for null or na vales 
insu.isna().sum()
insu.isnull().sum()


#####"Customer ID" is irrelevant and "Count",'Quarter' columns has no variance 
insu_1 = insu.drop(["Customer"], axis=1) # Id# is nothing just index  



########exploratory data analysis

EDA = {"columns_name ":insu_1.columns,
                  "mean":insu_1.mean(),
                  "median":insu_1.median(),
                  "mode":insu_1.mode(),
                  "standard_deviation":insu_1.std(),
                  "variance":insu_1.var(),
                  "skewness":insu_1.skew(),
                  "kurtosis":insu_1.kurt()}

EDA


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(insu_1.iloc[:, :])


#boxplot for every columns
insu_1.nunique()
boxplot = insu_1.boxplot(column=[ "Customer Lifetime Value","Customer Lifetime Value",
                                 "Monthly Premium Auto","Months Since Policy Inception","Total Claim Amount"])


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


#unique value for each columns 
insu_1.nunique()

insu_1 = insu_1.iloc[:, [1,8,11,12,13,20,0,2,3,4,5,6,7,9,10,14,15,16,17,18,19,21,22]]
insu_1.info()
# Normalized data frame (considering the numerical part of data)
insu_c_norm = norm_func(insu_1.iloc[:,0:6 ])
insu_c_norm.describe()

#########one hot encoding for discret data
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(insu_1.iloc[:,7:]).toarray())


insu_norm = pd.concat([insu_c_norm,enc_df], axis=1)

###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdis


TWSS = []
k = list(range(4, 12))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(insu_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 8  clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 7)
model.fit(insu_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
insu_1['clust7'] = mb # creating a  new column and assigning it to new column 


###########final data with  8 clusters 

insu_final = insu_1.iloc[:, [23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
insu_final.head()

# Aggregate mean of each cluster
fclust_details  = insu_final.iloc[:, 1:].groupby(insu_final.clust7).mean()
fclust_details


# creating a csv file  for new data frame with cluster 
insu_final.to_csv("insu_final.csv", encoding = "utf-8")

# creating a csv file  with details of each cluster 
fclust_details.to_csv("fclust_details.csv", encoding = "utf-8")