
library(readr)

insurance<-read_csv(file.choose())


summary(insurance)

# Normalize the data
normalized_data <- scale(insurance[, ]) # Excluding the nominal column

summary(normalized_data)


# Elbow curve to decide the k value
twss <- NULL
for (i in 2:7) {
  twss <- c(twss, kmeans(normalized_data, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:7, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")


# 4 Cluster Solution
fit <- kmeans(normalized_data, 4) 
str(fit)
fit$cluster



#creating a final data frame with cluster num 
final <- data.frame(fit$cluster,insurance)

#details of each group
fclust_details <-aggregate(final[,2:6], by = list(final$fit.cluster), FUN = mean)
fclust_details

#saving the result
write_csv(final, "hclustoutput.csv")


