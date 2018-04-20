library(foreign)
library(randomForest)
source("helper.R")

rf_prediction <- function(data_file, outcome, features=".", to_drop=c()){

	df <- read.dta(data_file)
	y_ <- factor(df[[outcome]])
	X_ <- drop_columns(df, c(to_drop, outcome))

	if(!(features == ".")){
		X_ <- X_[, features]
	}

	X_ <- data.frame(lapply(X_, factor))
	
	nonan_indices <- !is.na(y_)
	X <- X_[nonan_indices, ]
	y <- y_[nonan_indices]

	if(!(features == ".")){
		features <- paste(features, collapse="+")
	}

	forml <- as.formula(paste("outcome ~ ", features))
	rf <- randomForest(as.formula(forml), data=data.frame(X, outcome=y), ntree=500)

	predictions <- predict(rf, X_)
	list(model=rf, predictions=predictions)
}
