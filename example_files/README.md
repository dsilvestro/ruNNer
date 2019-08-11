## Format training input file

The training data will be split by default in a training+validation dataset and a test set (10% of the entries).
The training+validation dataset is split in training (80% of the entries) and validation (20%) and is used to 
determine the number of epochs yielding the highest validation accuracy. 

The test set is used after training the NN as an independent sample to quantify accuracy.

The format of the data is a simple tab-separated table without headers including 1 row per sample. 
The first column indicates the known classification of the sample and is given by an integer.
The other columns are the features that will be used by the NN to predict the label in the first column.

## Format empirical input file

This represents the input file for which classification is unknown and is inferred based on a trained NN.
The format is similar to that of the training data, but will only include the features, thus lacking the first column.
