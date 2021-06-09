# Load the dataset
library(readr)
insu <- read_csv(file.choose())
insu_1 <- insu[ , c(2:24)]

summary(insu_1)


library(fastDummies)

#columns name 
names(insu_1)

###### Normalization ###################################
# categorical data 
data_dummy  <- dummy_cols(insu_1, select_columns = c("State","Response"    ,   "Coverage" ,"Education"    ,  "Effective To Date","EmploymentStatus","Gender","Location Code" ,"Marital Status","Number of Open Complaints",  "Number of Policies","Policy Type","Policy", "Renew Offer Type" ,"Sales Channel"  ,"Vehicle Class","Vehicle Size"),
                          remove_first_dummy = TRUE,remove_most_frequent_dummy = FALSE,remove_selected_columns = TRUE)

# Normalize the data
# to normalize the data we use custom function 
norm <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

df_norm <- as.data.frame(lapply(data_dummy, norm)) # Excluding the nominal column




summary(df_norm)

# Elbow curve to decide the k value
twss <- NULL
for (i in 4:12) {
  twss <- c(twss, kmeans(df_norm, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(4:12, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")


# 8  Cluster Solution
fit <- kmeans(df_norm, 7) 
str(fit)
fit$cluster




#creating a final data frame with cluster num 
final <- data.frame(fit$cluster, insu_1)

#details of each group
fclust_details <-aggregate(insu_1[,c(2,9,12,13,14,15,16,21)], by = list(final$fit.cluster), FUN = mean)
fclust_details
#saving the result
write_csv(final, "hclustoutput.csv")


