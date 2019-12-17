# ruNNer -- Train and run neural networks in Python

General purpose program to train and run a neural network using keras and tensorflow (v.2) in Python (v.3) based on an array of numbers as input (features) and a discrete classification as output.

## Updated version to Tensorflow 2.0

### Train NN and get test accuracy
Run optimization for 500 epochs, using 2 hidden layers, number of nodes set equal to the number of features:

`python3 ruNNer.py -l example_files/example1_training_labels.txt -t example_files/example1_training_features.txt -verbose 0 -mode train -layers 2 -epochs 500`

The output of this command includes a trained NN and a pdf with plots for training and validation loss and accuracy. Reports accuracy of prediction and confusion matrix for test set (20% of input data).


### Use NN to predict
`python3 ruNNer.py -e example_files/example1_empirical_data.txt -loadNN example_files/example1_training_features_NN.NN -mode predict`
