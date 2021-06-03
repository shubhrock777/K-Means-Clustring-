# Load the dataset
library(readr)
crime <- read_csv(file.choose())
crime_1 <- crime[ , c(1:5)]

summary(crime_1)

# Normalize the data
normalized_data <- scale(crime_1[, 2:5]) # Excluding the nominal column

summary(normalized_data)

# Distance matrix dis= euclidean 
d <- dist(normalized_data, method = "euclidean") 

#linkage = complete
fit <- hclust(d, method = "complete")

# Display dendrogram
plot(fit) 
plot(fit, hang = -1)

groups <- cutree(fit, k = 4) # Cut tree into 3 clusters

rect.hclust(fit, k = 4, border = "blue")

membership <- as.matrix(groups)

#creating a final data frame with cluster num
final <- data.frame(membership, crime_1)

#details of each group
aggregate(crime_1[, 2:5], by = list(final$membership), FUN = mean)

#saving the result
write_csv(final, "hclustoutput.csv")

