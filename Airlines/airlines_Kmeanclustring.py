
import pandas as pd

import matplotlib.pylab as plt

air = pd.read_excel(r"D:\\BLR10AM\\Assi\\06Hierarchical clustering\\Dataset_Assignment Clustering\\EastWestAirlines.xlsx",sheet_name="data")


#######feature of the dataset to create a data dictionary
description  = ["Unique ID",
                "Number of miles eligible for award travel",
                "Number of miles counted as qualifying for Topflight status",
                 "Number of miles earned with freq. flyer credit card in the past 12 months:",
                 "Number of miles earned with Rewards credit card in the past 12 months:",
                 "Number of miles earned with Small Business credit card in the past 12 months: 1 = under 5,000 2 = 5,000 - 10,000 3 = 10,001 - 25,000  4 = 25,001 - 50,000 5 = over 50,000",
                  "Number of miles earned from non-flight bonus transactions in the past 12 months",
                 "Number of non-flight bonus transactions in the past 12 months",
                "Number of flight miles in the past 12 months",
                "Number of flight transactions in the past 12 months",
                "Number of days since Enroll_date",
                "Dummy variable for Last_award (1=not null, 0=null)"]

d_types =["count","ratio","continuous","count","count","count","continuous","continuous","continuous","continuous","continuous","count"]

data_details =pd.DataFrame({"column name":air.columns,
                "description":description,
                "data types ":d_types})


###########Data Pre-processing 

#unique value for each columns 
col_uni =air.nunique()

#details of dataframe
air.describe()
air.info()

#checking for null or na vales 
air.isna().sum()
air.isnull().sum()



air_1 = air.drop(["ID#"], axis=1) # Id# is nothing just index  



########exploratory data analysis

EDA = {"columns_name ":air_1.columns,
                  "mean":air_1.mean(),
                  "median":air_1.median(),
                  "mode":air_1.mode(),
                  "standard_deviation":air_1.std(),
                  "variance":air_1.var(),
                  "skewness":air_1.skew(),
                  "kurtosis":air_1.kurt()}

EDA


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(air_1.iloc[:, :])


#boxplot for every column
for column in air_1:
    plt.figure()
    air_1.boxplot([column])


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


#unique value for each columns 
air.nunique()

air_1 = air_1.iloc[:, [0,1,5,6,7,8,9,10,2,3,4]]
air_1.nunique()
# Normalized data frame (considering the numerical part of data)
air_c_norm = norm_func(air_1.iloc[:,0:7 ])
air_c_norm.describe()

#########one hot encoding for discret data
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(air_1.iloc[:,7:]).toarray())


air_norm = pd.concat([air_c_norm,enc_df], axis=1)

###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 


TWSS = []
k = list(range(3, 10))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(air_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(air_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
air_1['clust4'] = mb # creating a  new column and assigning it to new column 


###########final data with 4 clusters 

air_final = air_1.iloc[:, [11,0,1,2,3,4,5,6,7,8,9,10]]
air_final.head()

# Aggregate mean of each cluster
fclust_details  = air_final.iloc[:, 1:].groupby(air_final.clust4).mean()
fclust_details


# creating a csv file  for new data frame with cluster 
air_final.to_csv("air_final.csv", encoding = "utf-8")

# creating a csv file  with details of each cluster 
fclust_details.to_csv("fclust_details.csv", encoding = "utf-8")

