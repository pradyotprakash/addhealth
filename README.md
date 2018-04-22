This repository contains code to perform classification and predict the labels for a feature based on some covariates. The Add Health dataset for this has been included in the file `forestmissing.dta`. The code in `demo.R` shows how to use the functions.

Two predictors (based on random forests and feed forward neural networks) have been included as of now. These have been defined in the files `rf_prediction.R` and `nn_prediction.R`. The function definition is consistent across the two files and one needs to specify the:
- *data_file*: the Stata file containing the dataset
- *outcome*: an R string which is the outcome variable
- *features* (optional argument): a vector of the set of covariates to regress against. If this is not specified, all the features are included
- *to_drop* (optional argument): features which shouldn't be included in the model construction. If only a subset of the features are not to be used, then one can specify them using this argument and leave the *features* argument blank

The predictions of the various algorithms are stored in the variable *predictions* and the accuracy of each model is with respect to those data points which do not have a *NA* for the outcome feature.

### Adding algorithms
To extend the code to include more methods of performing classification, create a file with an appropriate name and import it file in `demo.R`. Add an entry in variable *functions* of the form *c(\<string describing the algorithm\>, \<function handle\>)* and it will be used for classification.

### Install random forest library
The random forest module can be installed by calling `install.packages("randomForest")` within R.

### Installing Tensorflow
The neural network model is constructed using the library tensorflow. <https://tensorflow.rstudio.com/tensorflow/articles/installation.html> explains how to install tensorflow on your machine. The tutorial talks about GPUs but we do not need them for our purposes.
