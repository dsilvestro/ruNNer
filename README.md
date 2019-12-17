# ruNNer -- Train and run neural networks in Python

General purpose program to train and run a neural network using keras and tensorflow (v.2) in Python (v.3) based on an array of numbers as input (features) and a discrete classification as output.

## Updated version to Tensorflow 2.0

### Train NN and get test accuracy
Run optimization for 500 epochs, using 2 hidden layers, number of nodes set equal to the number of features:

`python3 ruNNer.py -l example_files/example1_training_labels.txt -t example_files/example1_training_features.txt -verbose 0 -mode train -layers 2 -epochs 500`

The output of this command includes a trained NN and a pdf with plots for training and validation loss and accuracy. Reports accuracy of prediction and confusion matrix for test set (20% of input data).


### Use NN to predict
`python3 ruNNer.py -e example_files/example1_empirical_data.txt -loadNN example_files/example1_training_features_NN.NN -mode predict`



---
### Format training input file

The training data will be split by default in a training+validation dataset and a test set (10% of the entries).
The training+validation dataset is split in training (80% of the entries) and validation (20%) and is used to 
determine the number of epochs yielding the highest validation accuracy. 

The test set is used after training the NN as an independent sample to quantify accuracy.

The format of the data is a simple tab-separated table without headers including 1 row per sample. 
The first column indicates the known classification of the sample and is given by an integer.
The other columns are the features that will be used by the NN to predict the label in the first column.

### Format empirical input file

This represents the input file for which classification is unknown and is inferred based on a trained NN.
The format is similar to that of the training data, but will only include the features, thus lacking the first column.
