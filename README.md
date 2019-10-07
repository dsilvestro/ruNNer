# ruNNer -- Train and run neural networks in Python

General purpose script to train and run a neural network using keras and tensorflow in Python (v.3) based on an array of numbers as input (features) and a discrete classification as output.

## Example - training and testing a NN
`python3 ruNNer.py -mode train -layers 2 -t example_files/training_features.txt -l example_files/training_labels.txt -seed 1234 -outpath example_files/training_results/ -test 0.2`

This command saves a trained NN and a pdf with plots for training and validation loss and accuracy. Reports accuracy of prediction for test set (20% of input data).

## Example - running cross validation for NN
`python3 ruNNer.py -mode train -layers 2 -t example_files/training_features.txt -l example_files/training_labels.txt -seed 1234 -outpath example_files/training_results_cv/ -test 0.0 -cross_val 5`

This command runs a 5-fold cross-validation, at each time using different 20% of the training data array as validation data.

## Example - using NN to analyze data
`python3 ruNNer.py -mode predict -layers 2 -loadNN example_files/training_results/NN_2layers100epochs0batchreluglorot_normal_1234 -l example_files/training_labels.txt -seed 1234 -e example_files/empirical_data.txt -outpath example_files/model_predictions/`

This command saves two output files with the labels of the best output and probability for each category in the output.
