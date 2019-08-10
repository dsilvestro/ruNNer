# ruNNer
## Train and run neural networks in Python

General purpose script to train and run a neural network using keras and tensorflow in Python (v.3) based on an array of numbers as input (features) and a discrete classification as output.

## Example - training and testing a NN
`python3 ruNNer.py -train example_files/training_data.txt -layers 2`

This command saves a trained NN and a pdf with plots for training and validation loss and accuracy.


## Example - using NN to analyze data
`python3 ruNNer.py -loadNN example_files/testNN_2layers -layers 2 -data example_files/empirical_data.txt -outlabels red blue green yellow`

This command saves two output files with the labels of the best output and probability for each category in the output.
