
import pandas as pd 
import matplotlib.pyplot as plt
insurance= pd.read_csv("D:/BLR10AM/Assi/07Kmeans clustering/Datasets_Kmeans/Insurance Dataset.csv")



#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":insurance.columns,
                "data types ":insurance.dtypes})


###########Data Pre-processing 

#unique value for each columns 
col_uni =insurance.nunique()
col_uni


#details of dataframe
insurance.describe()
insurance.info()

#checking for null or na vales 
insurance.isna().sum()
insurance.isnull().sum()





########exploratory data analysis

EDA = {"columns_name ":insurance.columns,
                  "mean":insurance.mean(),
                  "median":insurance.median(),
                  "mode":insurance.mode(),
                  "standard_deviation":insurance.std(),
                  "variance":insurance.var(),
                  "skewness":insurance.skew(),
                  "kurtosis":insurance.kurt()}

EDA


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(insurance.iloc[:, :])


#boxplot for every column
insurance.columns
boxplot = insurance.boxplot(column=[ 'Premiums Paid', 'Age', 'Days to Renew', 'Claims made', 'Income'])


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)



# Normalized data frame (considering the numerical part of data)
insurance_norm = norm_func(insurance.iloc[ :,: ])
insurance_norm.describe()



###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans
TWSS = []
k = list(range(2,7))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(insurance_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(insurance_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
insurance['clust4'] = mb # creating a  new column and assigning it to new column 


###########final data with 8 clusters 

insurance_final = insurance.iloc[:, [5,0,1,2,3,4]]
insurance_final.head()

# Aggregate mean of each cluster
fclust_details  = insurance_final.iloc[:, 1:].groupby(insurance_final.clust4).mean()
fclust_details


# creating a csv file  for new data frame with cluster 
insurance_final.to_csv("insurance_final.csv", encoding = "utf-8")

# creating a csv file  with details of each cluster 
fclust_details.to_csv("fclust_details.csv", encoding = "utf-8")
