# Load the dataset
library(readxl)
tel <- read_excel(file.choose())

# droping theis columns "Customer ID","Count",'Quarter'
tel_1 <- tel[ , c(4:30)]

summary(tel_1)


library(fastDummies)

#columns name 
names(tel_1)

###### Normalization ###################################
# categorical data 
data_dummy  <- dummy_cols(tel_1, select_columns = c("Referred a Friend","Response" , "Offer" , 
                                                    "Phone Service" ,"Multiple Lines", "Internet Service"                 
                                                    ,"Internet Type","Online Security" , "Online Backup"                    
                                                    ,"Device Protection Plan" ,"Premium Tech Support", "Streaming TV", "Streaming Movies"                 
                                                    , "Streaming Music" ,"Unlimited Data" , "Contract" ,"Paperless Billing"                
                                                    ,"Payment Method"  ),
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
for (i in 3:12) {
  twss <- c(twss, kmeans(df_norm, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(3:12, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")


# 8 Cluster Solution
fit <- kmeans(df_norm, 8) 
str(fit)
fit$cluster

#creating a final data frame with cluster num 
final <- data.frame(fit$cluster,tel_1)

#details of each group
fclust_details <-aggregate(tel_1[,c(3,6,10,22,23,24,25,26,27)], by = list(final$fit.cluster), FUN = mean)
fclust_details
#saving the result
write_csv(final, "hclustoutput.csv")

