# python3
# Created by Daniele Silvestro on 2019.05.23
import keras, os
import numpy as np
from numpy import *
import scipy.special
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
from keras.models import Sequential
from keras.layers import Dense
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import set_random_seed
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend
import argparse, sys

p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-train',           type=str,   help='training data set', default= "", metavar= "")
p.add_argument('-test',            type=float, help='fraction of training used as test set', default= 0.1, metavar= 0.1)
p.add_argument('-data',            type=str,   help='empirical_data  ',default= "", metavar= "")
p.add_argument('-r',               type=str,   help='scaling_array ',  default= "", metavar= "")
p.add_argument('-outlabels',       type=str,   nargs='+',default=[])
p.add_argument('-layers',          type=int,   help='n. hidden layers', default= 1, metavar= 1)
p.add_argument('-path',            type=str,   help='', default= "", metavar= "")
p.add_argument('-batch_size',      type=int,   help='if 0: dataset is not sliced into smaller batches', default= 0, metavar= 0)
p.add_argument('-epochs',          type=int,   help='', default= 100, metavar= 100)
p.add_argument('-verbose',         type=int,   help='', default= 1, metavar= 1)
p.add_argument('-loadNN',          type=str,   help='', default= '', metavar= '')
p.add_argument('-seed',            type=int,   help='', default= 0, metavar= 0)
p.add_argument('-actfunc',         type=int,   help='1) relu; 2) tanh; 3) sigmoid', default= 1, metavar= 1)
p.add_argument('-kerninit',        type=int,   help='1) glorot_normal; 2) glorot_uniform', default= 1, metavar= 1)
p.add_argument('-nodes',           type=float, help='n. nodes (multiplier of n. features)', default= 1, metavar= 1)
p.add_argument('-randomize_data',  type=float, help='shuffle order data entries', default= 1, metavar= 1)
p.add_argument('-threads',         type=int,   help='n. of threads (0: system picks an appropriate number)', default= 0, metavar= 0)
args = p.parse_args()

# NN SETTINGS
n_hidden_layers = args.layers # number of extra hidden layers
max_epochs = args.epochs
batch_size_fit = args.batch_size # batch size
units_multiplier = args.nodes # number of nodes per input 
plot_curves = 1 
train_nn = 1
run_test_accuracy = 0 
run_empirical = 1
run_tests = 0
randomize_data = args.randomize_data

# SET SEEDS
if args.seed==0: rseed = np.random.randint(1000,9999)
else: rseed = args.seed
np.random.seed(rseed)
random.seed(rseed)
set_random_seed(rseed)
n_threads = args.threads
session_conf = tf.ConfigProto(intra_op_parallelism_threads=n_threads, inter_op_parallelism_threads=n_threads)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)


activation_functions = ["relu", "tanh", "sigmoid"]
activation_function = activation_functions[args.actfunc-1]

kernel_initializers = ["glorot_normal", "glorot_uniform"]
kernel_init = kernel_initializers[args.kerninit-1]

path = args.path

if args.loadNN == "":
	model_name = "%sNN_%slayers%sepochs%sbatch%s%s_%s" % (path, n_hidden_layers,max_epochs,batch_size_fit,activation_function,kernel_init,rseed)
else:
	model_name = args.loadNN

# input files
file_empirical_data  = args.data # empirical data 
file_scaling_array   = args.r # rescale features 

# process train dataset
if args.train != "":
	try:
		input_data = np.loadtxt(args.train) # load txt file
	except: 
		input_data = np.load(args.train) # load npy file
	train_indx = range( int(input_data.shape[0]*(1-args.test)) )
	input_training = input_data[train_indx,1:]
	input_trainLabels = input_data[train_indx,0].astype(int)
	input_trainLabelsPr = np.zeros((len(input_training[train_indx,0]), len(np.unique(input_trainLabels))) )
	j =0
	for i in np.sort(np.unique(input_trainLabels)):
		input_trainLabelsPr[input_trainLabels==i,j]=1
		j+=1
	if batch_size_fit==0:
		batch_size_fit = int(input_training.shape[0])
	train_nn = 1
	test_nn  = 1
else: 
	train_nn = 0
	test_nn  = 0

if file_scaling_array !="":
	scaling_array       = np.loadtxt(file_scaling_array, skiprows=1)
else:
	scaling_array = 1.
size_output         = len(args.outlabels)
print(size_output)

if train_nn:		
	input_training = input_training / scaling_array
	
	# DEF SIZE OF THE FEATURES
	hSize = np.shape(input_training)[1]
	nCat  = np.shape(input_trainLabelsPr)[1]
	
	# size of dataset
	dSize = np.shape(input_training)[0]
	if randomize_data:
		rnd_indx = np.random.choice(np.arange(dSize),dSize,replace = False)
		# shuffle data
		input_training = input_training[rnd_indx,:]
		input_trainLabels = input_trainLabels[rnd_indx]
		input_trainLabelsPr = input_trainLabelsPr[rnd_indx,:]

	modelFirstRun=Sequential() # init neural network
	### DEFINE from INPUT HIDDEN LAYER
	modelFirstRun.add(Dense(input_shape=(hSize,),units=int(units_multiplier*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
	### ADD HIDDEN LAYER
	for jj  in range(n_hidden_layers-1):
		modelFirstRun.add(Dense(units=int(units_multiplier*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
	
	modelFirstRun.add(Dense(units=nCat,activation="softmax",kernel_initializer=kernel_init,use_bias=True))
	modelFirstRun.summary()
	modelFirstRun.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
	print("Running model.fit") 
	history=modelFirstRun.fit(input_training,input_trainLabelsPr,epochs=max_epochs,batch_size=batch_size_fit,validation_split=0.2,verbose=args.verbose)

	if plot_curves:
		fig = plt.figure(figsize=(20, 8))
		fig.add_subplot(121)
		plt.plot(history.history['loss'],'r',linewidth=3.0)
		plt.plot(history.history['val_loss'],'b',linewidth=3.0)
		plt.legend(['Training loss', 'Validation Loss'],fontsize=12)
		plt.xlabel('Epochs',fontsize=12)
		plt.ylabel('Loss',fontsize=12)
		plt.title('Loss Curves',fontsize=12)
 
		# Accuracy Curves
		fig.add_subplot(122)
		plt.plot(history.history['acc'],'r',linewidth=3.0)
		plt.plot(history.history['val_acc'],'b',linewidth=3.0)
		plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=12)
		plt.xlabel('Epochs',fontsize=12)
		plt.ylabel('Accuracy',fontsize=12)
		plt.title('Accuracy Curves',fontsize=12)
		
		file_name = "%s_res.pdf" % (model_name)
		pdf = matplotlib.backends.backend_pdf.PdfPages(file_name)
		pdf.savefig( fig )
		pdf.close()
	
	# OPTIM OVER VALIDATION AND THEN TEST ON TEST DATASET (THAT'S THE FINAL ACCURACY)
	optimal_number_of_epochs = np.argmin(history.history['val_loss'])
	print("optimal number of epochs:", optimal_number_of_epochs+1)
	history.history['val_acc'][optimal_number_of_epochs]

	model=Sequential() # init neural network
	model.add(Dense(input_shape=(hSize,),units=int(units_multiplier*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
	for jj in range(n_hidden_layers-1):
		model.add(Dense(units=int(units_multiplier*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
	model.add(Dense(units=nCat,activation="softmax",kernel_initializer=kernel_init,use_bias=True))
	model.summary()
	model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
	history=model.fit(input_training,input_trainLabelsPr,epochs=optimal_number_of_epochs+1,batch_size=batch_size_fit,validation_split=0.2, verbose=args.verbose)	
	model.save_weights(model_name)
	print("\nModel saved as:", model_name, optimal_number_of_epochs+1)


if test_nn:
	test_indx = range( int(input_data.shape[0]*(1-args.test)), input_data.shape[0] )
	print(test_indx)
	input_test = input_data[test_indx,1:]
	print(input_test.shape)
	input_testLabels = input_data[test_indx,0].astype(int)
	print(input_testLabels.shape)
	input_testLabelsPr = np.zeros((input_test.shape[0], len(np.unique(input_testLabels))) )
	j =0
	for i in np.sort(np.unique(input_testLabels)):
		input_testLabelsPr[input_testLabels==i,j]=1
		j+=1
	
	
	predictions=np.argmax(model.predict(input_test),axis=1)
	confusion_matrix(input_testLabels,predictions)
	scores=model.evaluate(input_test,input_testLabelsPr,verbose=0)
	print("\nTest accuracy rate: %.2f%%"%(scores[1]*100))
	print("Test error rate: %.2f%%"%(100-scores[1]*100))
	print('Test cross-entropy loss:',round(scores[0],3),"\n")


######## TEST EMPIRICAL DATA SETS
if file_empirical_data !="":
	print("Loading input file...")
	input_data = np.loadtxt(file_empirical_data)
	input_data = input_data / scaling_array
	print("Loading weights...")
	# DEF SIZE OF THE FEATURES
	hSize = input_data.shape[1]
	model=Sequential() # init neural network
	model.add(Dense(input_shape=(hSize,),units=int(units_multiplier*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
	for jj  in range(n_hidden_layers-1):
		model.add(Dense(units=int(units_multiplier*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
	model.add(Dense(units=size_output,activation="softmax",kernel_initializer=kernel_init,use_bias=True))
	model.summary()
	model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
	model.load_weights(model_name, by_name=False)
	print("done.")
	print(np.shape(input_data))

	estimate_par = model.predict(input_data)
	print(estimate_par)
	f_name = os.path.basename(file_empirical_data)
	f_name = os.path.splitext(f_name)[0]
	model_name ="%slayers%sepochs%sbatch%s%s_%s" % (n_hidden_layers,max_epochs,batch_size_fit,activation_function,kernel_init,rseed)
	outfile = "%sNN_prob.txt" % (f_name)
	np.savetxt(outfile, np.round(estimate_par,4), delimiter="\t",fmt='%1.4f')
	
	lab = np.array(args.outlabels)
	indx_best = argmax(estimate_par,axis=1)
	print(sum(indx_best))
	outfile = "%s_NN_labels.txt" % (f_name)
	np.savetxt(outfile, lab[indx_best], delimiter="\t",fmt="%s")
