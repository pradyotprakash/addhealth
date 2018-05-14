library(foreign)
source("rf_prediction.R")
source("nn_prediction.R")

functions <- list(c("rf", rf_prediction), c("nn", nn_prediction))
features <- "."
outcome <- "selldrugsw1"
to_drop <- c("id", "_merge", "anymarijuanaw1", "hopefulfuturew1")

data_file <- "forestmissing.dta"
predictions <- list()

df <- read.dta(data_file)
y <- factor(df[[outcome]])
nonan_indices <- !is.na(y)

for(func in functions){
	v <- func[[2]](data_file, outcome, features=features, to_drop=to_drop)
	predictions[[func[[1]]]] <- v$predictions
	cat("Accuracy of", func[[1]], ":", sum(y[nonan_indices] == v$predictions[nonan_indices])/sum(nonan_indices), "\n")
}
