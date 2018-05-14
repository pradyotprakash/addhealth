library(foreign)
library(tensorflow)
source("helper.R")

getWeights <- function(widths){
	l <- c()
	u <- c()
	
	for(i in 1:(length(widths) - 1)){
		w <- tf$Variable(tf$random_normal(shape(widths[i], widths[i+1]), stddev=0.01))
		b <- tf$Variable(tf$random_normal(shape(1, widths[i+1]), stddev=0.01))
		l <- c(l, w)
		u <- c(u, b)
	}

	list(weights=l, biases=u)
}

getModel <- function(x, widths, probInput, probHidden){
	ret <- getWeights(widths)
	weights <- ret$weights
	biases <- ret$biases

	output <- tf$nn$dropout(x, probInput)
	for(i in 1:(length(weights)-1)){
		h <- tf$nn$relu(tf$matmul(output, weights[[i]]) + biases[[i]])
		output <- tf$nn$dropout(h, probHidden)
	}

	lastIndex <- length(weights)
	tf$matmul(output, weights[[lastIndex]])
}

get_integer_for_factor <- function(fact, dat){
	l <- list()
	labels <- levels(fact)

	for(i in 1:length(labels)){
		l[[labels[i]]] <- i
	}

	y = vector(length=length(dat))
	for(i in 1:length(dat)){
		y[i] <- l[[dat[i]]]
	}

	list(l=l, dat=y)
}

get_factor_for_integer <- function(mapping, dat){
	l <- list()
	for(i in names(mapping)){
		l[[mapping[[i]]]] <- i
	}

	y <- vector(length=length(dat))
	for(i in 1:length(dat)){
		y[i] <- l[[as.integer(dat[i])]]
	}
	factor(y)
}

nn_prediction <- function(data_file, outcome, features=".", to_drop=c()){

	df <- read.dta(data_file)
	y_ <- factor(df[[outcome]])
	X_ <- drop_columns(df, c(to_drop, outcome))

	if(!(features == ".")){
		X_ <- X_[, features]	
	}

	X_ <- data.frame(lapply(X_, factor))
	nonan_indices <- !is.na(y_)

	# one-hot encode the input vectors
	X1 <- model.matrix(~ . + 0, data=X_, contrasts.arg=lapply(X_, contrasts, contrasts=FALSE))
	X <- X1[nonan_indices, ]

	mapping <- get_integer_for_factor(y_, y_[nonan_indices])
	y <- mapping$dat

	labels_ = matrix(0, nrow(X), 2)
	for(i in 1:nrow(X)){
		labels_[i, y[i]] = 1.0
	}

	input_d <- ncol(X)
	output_d <- length(levels(y_))
	widths <- c(input_d, 150, 40, output_d)

	xx <- tf$placeholder(tf$float32, shape(NULL, input_d))
	labels <- tf$placeholder(tf$float32, shape(NULL, output_d))
	probInput <- tf$placeholder(tf$float32)
	probHidden <- tf$placeholder(tf$float32)

	yy <- getModel(xx, widths, probInput, probHidden)
	loss <- tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits=yy, labels=labels))
	trainStep <- tf$train$RMSPropOptimizer(0.001)$minimize(loss)
	predict <- tf$argmax(yy, 1L)

	sess <- tf$Session()
	sess$run(tf$global_variables_initializer())

	batchSize <- 100
	numBatches <- as.integer(nrow(X) / batchSize)
	for(i in 1:10){
		for(j in 1:numBatches){
			xs <- X[((j-1)*batchSize+1):((j)*batchSize), ]
			ys <- labels_[((j-1)*batchSize+1):((j)*batchSize), ]
			sess$run(trainStep, feed_dict=dict(xx=xs, labels=ys, probInput=1.0, probHidden=1.0))
		}
	}

	y_new <- 1 + sess$run(predict, feed_dict=dict(xx=X1, probInput=1.0, probHidden=1.0))
	predictions <- get_factor_for_integer(mapping$l, y_new)
	list(model=NA, predictions=predictions)
}
