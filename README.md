# ruNNer -- Train and run neural networks in Python

General purpose script to train and run a neural network using keras and tensorflow in Python (v.3) based on an array of numbers as input (features) and a discrete classification as output.

## Example - training and testing a NN
`python3 ruNNer.py -mode train -layers 2 -t example_files/training_features.txt -l example_files/training_labels.txt -seed 1234 -outpath example_files/training_results/`

This command saves a trained NN and a pdf with plots for training and validation loss and accuracy.


## Example - using NN to analyze data
`python3 ruNNer.py -mode predict -layers 2 -loadNN example_files/training_results/NN_1layers100epochs0batchreluglorot_normal_1234 -t example_files/training_features.txt -l example_files/training_labels.txt -seed 1234 -e example_files/empirical_data.txt -outpath example_files/model_predictions/`

This command saves two output files with the labels of the best output and probability for each category in the output.
