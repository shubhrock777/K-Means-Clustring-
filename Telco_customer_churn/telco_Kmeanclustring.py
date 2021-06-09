

import pandas as pd

import matplotlib.pyplot as plt

tel = pd.read_excel(r"D:/BLR10AM/Assi/06Hierarchical clustering/Dataset_Assignment Clustering/Telco_customer_churn.xlsx")


#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":tel.columns,
                "data types ":tel.dtypes})


###########Data Pre-processing 

#unique value for each columns 
col_uni =tel.nunique()
col_uni


#details of dataframe
tel.describe()
tel.info()

#checking for null or na vales 
tel.isna().sum()
tel.isnull().sum()


#####"Customer ID" is irrelevant and "Count",'Quarter' columns has no variance 
tel_1 = tel.drop(["Customer ID","Count",'Quarter'], axis=1) # Id# is nothing just index  



########exploratory data analysis

EDA = {"columns_name ":tel_1.columns,
                  "mean":tel_1.mean(),
                  "median":tel_1.median(),
                  "mode":tel_1.mode(),
                  "standard_deviation":tel_1.std(),
                  "variance":tel_1.var(),
                  "skewness":tel_1.skew(),
                  "kurtosis":tel_1.kurt()}

EDA


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(tel_1.iloc[:, :])


#boxplot for every column
col_uni
boxplot = tel_1.boxplot(column=[ 'Number of Referrals', 'Tenure in Months', 'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download','Monthly Charge',
                                'Total Charges','Total Refunds','Total Long Distance Charges','Total Revenue'])


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


#unique value for each columns 
tel_1.nunique()

tel_1 = tel_1.iloc[:, [2,5,9,21,22,23,24,25,26,0,1,3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20]]

# Normalized data frame (considering the numerical part of data)
tel_c_norm = norm_func(tel_1.iloc[:,0:9 ])
tel_c_norm.describe()

#########one hot encoding for discret data
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(tel_1.iloc[:,9:]).toarray())


tel_norm = pd.concat([tel_c_norm,enc_df], axis=1)


###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans
TWSS = []
k = list(range(4, 12))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(tel_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 8 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 8)
model.fit(tel_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
tel_1['clust8'] = mb # creating a  new column and assigning it to new column 


###########final data with 8 clusters 

tel_final = tel_1.iloc[:, [27,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]
tel_final.head()

# Aggregate mean of each cluster
fclust_details  = tel_final.iloc[:, 1:].groupby(tel_final.clust8).mean()
fclust_details


# creating a csv file  for new data frame with cluster 
tel_final.to_csv("tel_final.csv", encoding = "utf-8")

# creating a csv file  with details of each cluster 
fclust_details.to_csv("fclust_details.csv", encoding = "utf-8")

