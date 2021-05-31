# Load the dataset
library(readxl)
air <- read_excel("D:/BLR10AM/Assi/06Hierarchical clustering/Dataset_Assignment Clustering/EastWestAirlines.xlsx", sheet = 2)

# droping theis columns "Customer ID","Count",'Quarter'
air_1 <- air[ , c(2: 12)]

summary(air_1)


library(fastDummies)

#columns name 
names(air_1)

###### Normalization ###################################
# categorical data 
data_dummy  <- dummy_cols(air_1, select_columns = c("cc1_miles" , "cc2_miles" ,
                                                    "cc3_miles"  ," Flight_trans_12"   ,"Award?     "),
                          remove_first_dummy = TRUE,remove_most_frequent_dummy = FALSE,remove_selected_columns = TRUE)




# Normalize the data
# to normalize the data we use custom function
#'Total Charges','Total Refunds','Total Long Distance Charges','Total Revenue'
norm <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

df_norm <- as.data.frame(lapply(data_dummy, norm)) # Excluding the nominal column




summary(df_norm)


# Elbow curve to decide the k value
twss <- NULL
for (i in 2:9) {
  twss <- c(twss, kmeans(df_norm, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:9, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")


# 4 Cluster Solution
fit <- kmeans(df_norm, 4) 
str(fit)
fit$cluster






#creating a final data frame with cluster num 
final <- data.frame(fit$cluster,air_1)

#details of each group
fclust_details <-aggregate(final[,], by = list(final$fit.cluster), FUN = mean)
fclust_details

#saving the result
write_csv(final, "hclustoutput.csv")










