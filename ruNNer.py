# python3
# Created by Daniele Silvestro on 2019.05.23
import matplotlib
matplotlib.use('Agg')
import keras, os
import numpy as np
from numpy import *
import scipy.special
np.set_printoptions(suppress= 1) # prints floats, no scientific notation
np.set_printoptions(precision=3) # rounds all array elements to 3rd digit
from keras.models import Sequential
from keras.layers import Dense
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow import set_random_seed
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend
import argparse, sys


p = argparse.ArgumentParser() #description='<input file>') 
p.add_argument('-mode',                   choices=['train', 'predict'],default=None,required=True)
p.add_argument('-t',                      type=str,   help='array of training features',default= "", metavar= "")
p.add_argument('-l',                      type=str,   help='array of training labels', default= 0, metavar= 0)
p.add_argument('-e',                      type=str,   help='array of empirical features', default= 0, metavar= 0)
p.add_argument('-r',                      type=str,   help='file with rescaling array or float', default= 1, metavar= 1)
p.add_argument('-feature_indices',        type=str,   help='array of feature indices to select', default= 0, metavar= 0)
p.add_argument('-train_instance_indices', type=str,   help='array of indices for selecting training instances', default= 0, metavar= 0)
p.add_argument('-test',                   type=float, help='fraction of training used as test set', default= 0.1, metavar= 0.1)
p.add_argument('-n_labels',               type=int,   help='provide number of labels, necessary for prediction', default=0)
p.add_argument('-outlabels',              type=str,   nargs='+',default=[])
p.add_argument('-layers',                 type=int,   help='n. hidden layers', default= 1, metavar= 1)
p.add_argument('-outpath',                type=str,   help='', default= "")
p.add_argument('-outname',                type=str,   help='', default= "")
p.add_argument('-batch_size',             type=int,   help='if 0: dataset is not sliced into smaller batches', default= 0, metavar= 0)
p.add_argument('-epochs',                 type=int,   help='', default= 100, metavar= 100)
p.add_argument('-optim_epoch',            type=int,   help='0: min loss function; 1: max validation accuracy',default=0)
p.add_argument('-verbose',                type=int,   help='', default= 1, metavar= 1)
p.add_argument('-loadNN',                 type=str,   help='', default= '', metavar= '')
p.add_argument('-seed',                   type=int,   help='', default= 0, metavar= 0)
p.add_argument('-actfunc',                type=int,   help='1) relu; 2) tanh; 3) sigmoid', default= 1, metavar= 1)
p.add_argument('-kerninit',               type=int,   help='1) glorot_normal; 2) glorot_uniform', default= 1, metavar= 1)
p.add_argument('-nodes',                  type=float, help='n. nodes (multiplier of n. features)', nargs='+',default=[1.])
p.add_argument('-randomize_data',         type=float, help='shuffle order data entries', default= 1, metavar= 1)
p.add_argument('-threads',                type=int,   help='n. of threads (0: system picks an appropriate number)', default= 0, metavar= 0)
p.add_argument('-cross_val',              type=int,   help='Set number of cross validations to run. Set to 0 to turn off.',default=0)
p.add_argument('-validation_off',         action="store_true",help='If flag is used, no validation set will be used when training the model. Instead training will run until maximum number of epochs set with "-epochs" flag.',default=False)
args = p.parse_args()

# NN SETTINGS
n_hidden_layers = args.layers # number of extra hidden layers
max_epochs = args.epochs
batch_size_fit = args.batch_size # batch size
units_multiplier = args.nodes # number of nodes per input 
if n_hidden_layers != len(units_multiplier):
	units_multiplier = np.repeat(units_multiplier[0], n_hidden_layers)
	print("Using node multiplier:",units_multiplier)
plot_curves = 1 
train_nn = 1
randomize_data = args.randomize_data
#run_test_accuracy = 0
#run_tests = 0
if args.mode == 'train':
	run_train = 1
	run_empirical = 0
elif args.mode == 'predict':
	run_train = 0
	run_empirical = 1

try:
	rescale_factors = float(args.r)
except(ValueError):
	rescale_factors = np.loadtxt(args.r)

# SET SEEDS
if args.seed==0: rseed = np.random.randint(1000,9999)
else: rseed = args.seed
np.random.seed(rseed)
np.random.seed(rseed)
set_random_seed(rseed)

n_threads = args.threads
session_conf = tf.ConfigProto(intra_op_parallelism_threads=n_threads, inter_op_parallelism_threads=n_threads)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)

activation_functions = ["relu", "tanh", "sigmoid"]
activation_function = activation_functions[args.actfunc-1]

kernel_initializers = ["glorot_normal", "glorot_uniform"]
kernel_init = kernel_initializers[args.kerninit-1]

outpath = args.outpath
if outpath=="":
	outpath = os.path.dirname(args.t)
elif not os.path.exists(outpath):
 	os.makedirs(outpath)

if args.loadNN == "":
	if args.cross_val > 1:
		model_out = os.path.join(outpath,'cv')
		if not os.path.exists(model_out):
			os.makedirs(model_out)
	else:
		model_out = outpath
	model_name = os.path.join(model_out,"trained_model_NN_%slayers%sepochs%sbatch%s%s_%s" % (n_hidden_layers,max_epochs,batch_size_fit,activation_function,kernel_init,rseed))
else:
	model_name = args.loadNN

# input files
file_training_data  = args.t # empirical data 
file_empirical_data  = args.e # empirical data 
file_training_labels = args.l # training labels

# if labels are provided read them, because they will be used by training or prediction mode
if file_training_labels:
	try:
		training_labels = np.loadtxt(file_training_labels) # load txt file
	except: 
		training_labels = np.load(file_training_labels) # load npy file

# process train dataset
train_nn = 0
test_nn  = 0
if run_train:
	try:
		training_features = np.loadtxt(file_training_data) # load txt file
	except: 
		training_features = np.load(file_training_data) # load npy file
	training_features/=rescale_factors

	# scale data using the min-max scaler (between 0 and 1)
	scaler = MinMaxScaler()
	scaler.fit(training_features)
	training_features = scaler.transform(training_features)

	# select features and instances, if files provided:
	if args.feature_indices:
		feature_index_array = np.loadtxt(args.feature_indices,dtype=int)
		training_features = training_features[:,feature_index_array]
	if args.train_instance_indices:
		instance_index_array = np.loadtxt(args.train_instance_indices,dtype=int)
		training_features = training_features[instance_index_array,:]
		training_labels = training_labels[instance_index_array]

		
	dSize = np.shape(training_features)[0]
	if randomize_data:
		rnd_indx = np.random.choice(np.arange(dSize),dSize,replace = False)
		# shuffle data
		training_features = training_features[rnd_indx,:] + 0
		training_labels = training_labels[rnd_indx]+0
	
	
	# split into training and test set
	test_indx = range( int(training_features.shape[0]*(1-args.test)), training_features.shape[0] )
	#print(test_indx)
	input_test = training_features[test_indx,:]
	#print(input_test)
	input_testLabels = training_labels[test_indx].astype(int)
	input_testLabelsPr = np.zeros((input_test.shape[0], len(np.unique(input_testLabels))) )
	j = 0
	for i in np.sort(np.unique(input_testLabels)):
		input_testLabelsPr[input_testLabels==i,j]=1
		j+=1

	train_indx = range( int(training_features.shape[0]*(1-args.test)) )
	input_training = training_features[train_indx,:]
	input_trainLabels = training_labels[train_indx].astype(int)
	input_trainLabelsPr = np.zeros((len(input_training[train_indx,0]), len(np.unique(input_trainLabels))) )
	j = 0
	for i in np.sort(np.unique(input_trainLabels)):
		input_trainLabelsPr[input_trainLabels==i,j]=1
		j+=1
	if batch_size_fit==0:
		batch_size_fit = int(input_training.shape[0])
	
	print('Training data shape:', input_training.shape)

	train_nn = 1
	test_nn  = 1
	
	# DEF SIZE OF THE FEATURES
	hSize = np.shape(input_training)[1]
	nCat  = np.shape(input_trainLabelsPr)[1]
	dSize = np.shape(input_training)[0]


	if args.cross_val > 1:
		training_data = []
		training_labels = []
		validation_data_list = []
		skf = StratifiedKFold(n_splits=int(args.cross_val))
		for train, test in skf.split(input_training,input_trainLabelsPr[:,0]):
			training_data.append(input_training[train,:])
			training_labels.append(input_trainLabelsPr[train,:])
			validation_features = input_training[test,:]
			validation_labels = input_trainLabelsPr[test,:]
			validation_data_list.append((validation_features,validation_labels))
	elif args.validation_off:
		training_data = [input_training]
		training_labels = [input_trainLabelsPr]
		validation_features = []
		validation_labels = []
		validation_data_list = [(validation_features, validation_labels)]
	else:
		index = int(input_training.shape[0]*0.8)
		training_data = [input_training[:index,:]]
		training_labels = [input_trainLabelsPr[:index,:]]
		validation_features = input_training[index:,:]
		validation_labels = input_trainLabelsPr[index:,:]
		validation_data_list = [(validation_features, validation_labels)]
      
	accuracy_scores = []
	best_epochs = []
	for i,input_training in enumerate(training_data):
		input_trainLabelsPr = training_labels[i]
		validation_data = validation_data_list[i]
		modelFirstRun=Sequential() # init neural network
		### DEFINE from INPUT HIDDEN LAYER
		modelFirstRun.add(Dense(input_shape=(hSize,),units=int(units_multiplier[0]*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
		### ADD HIDDEN LAYER
		for jj  in range(n_hidden_layers-1):
			modelFirstRun.add(Dense(units=int(units_multiplier[jj+1]*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
		
		modelFirstRun.add(Dense(units=nCat,activation="softmax",kernel_initializer=kernel_init,use_bias=True))
		modelFirstRun.summary()
		modelFirstRun.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
		print("Running model.fit")
      # if no validation data (set by user) just train until final epoch
		if len(validation_data[0]) == 0:
			history=modelFirstRun.fit(input_training,input_trainLabelsPr,epochs=max_epochs,batch_size=batch_size_fit,verbose=args.verbose)
			model = modelFirstRun
			print("Running training without validation set")
		else: 
			history=modelFirstRun.fit(input_training,input_trainLabelsPr,epochs=max_epochs,batch_size=batch_size_fit,validation_data=validation_data,verbose=args.verbose)

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
				
				#file_name = "%s_res.pdf" % (model_name.replace('trained_model_',''))
				#pdf = matplotlib.backends.backend_pdf.PdfPages(file_name)
				#pdf.savefig( fig )
				#pdf.close()

			# OPTIM OVER VALIDATION AND THEN TEST ON TEST DATASET (THAT'S THE FINAL ACCURACY)
			if args.optim_epoch==0:
				optimal_number_of_epochs = np.argmin(history.history['val_loss'])
			elif args.optim_epoch==1:
				optimal_number_of_epochs = np.argmax(history.history['val_acc'])
			best_epochs.append(optimal_number_of_epochs)
			print("optimal number of epochs:", optimal_number_of_epochs+1)
			# print loss and accuracy at best epoch to file
			loss_at_best_epoch = history.history['val_loss'][optimal_number_of_epochs]
			accurracy_at_best_epoch = history.history['val_acc'][optimal_number_of_epochs]
			model=Sequential() # init neural network
			model.add(Dense(input_shape=(hSize,),units=int(units_multiplier[0]*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
			for jj in range(n_hidden_layers-1):
				model.add(Dense(units=int(units_multiplier[jj+1]*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
			model.add(Dense(units=nCat,activation="softmax",kernel_initializer=kernel_init,use_bias=True))
			model.summary()
			model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
			history=model.fit(input_training,input_trainLabelsPr,epochs=optimal_number_of_epochs+1,batch_size=batch_size_fit, validation_data=validation_data, verbose=args.verbose)	

			accuracy = history.history['acc'][-1]
			accuracy_scores.append(np.round(accuracy,6))

		if args.cross_val > 1:
			weight_file_name = model_name+'_cv_%i'%i
		else:
			weight_file_name = model_name
		model.save_weights(weight_file_name)
		print("Model saved as:", weight_file_name)

	try:
		# plot all accuracy curves (multiple pages in case of cv)
		file_name = "%s_res.pdf" % (model_name.replace('trained_model_',''))
		pdf = matplotlib.backends.backend_pdf.PdfPages(file_name)
		for figure in range(1, fig.number + 1):
			pdf.savefig(figure)
		pdf.close()
		plt.close('all')
	except:
		no_plot=True

	# write output text file
	info_out = os.path.join(outpath,'info.txt')
	args_data = vars(args)
   # adjust seed since it may have been randomely drawn
	args_data['seed'] = rseed
	# add the shape of the training data input
	args_data['total_training_array_shape'] = str(input_training.shape)

	print('Putting away %.3f of the data as test set. Dimensions of resulting test set: %s.'%(args.test,str(input_test.shape)))

	if not args.validation_off:
	   # add the list of best epochs and accuracies to output df
		args_data['best_epoch'] = str(best_epochs)
		args_data['accuracies'] = str((accuracy_scores))	
		print('Best epoch (average):', int(np.round(np.mean(best_epochs))))
		print('Validation accuracy (average):', np.mean(accuracy_scores))

	with open(info_out,"w") as f:
		for i in args_data:
			f.write(f"{i}\t{str(args_data[i])}\n") 
      

if test_nn and args.test > 0. and not args.cross_val > 1:	
	predictions=np.argmax(model.predict(input_test),axis=1)
	confusion_matrix(input_testLabels,predictions)
	scores=model.evaluate(input_test,input_testLabelsPr,verbose=0)
	print("\nTest accuracy rate: %.2f%%"%(scores[1]*100))
	print("Test error rate: %.2f%%"%(100-scores[1]*100))
	print('Test cross-entropy loss:',round(scores[0],3),"\n")



######## TEST EMPIRICAL DATA SETS
if run_empirical:   
	print("Loading input file...")
	try:
		empirical_features = np.loadtxt(file_empirical_data)
	except:
		empirical_features = np.load(file_empirical_data)
		
	empirical_features/=rescale_factors    
	print(np.amin(empirical_features,0))
	print(np.amax(empirical_features,0))
	# select features and instances, if files provided:
	if args.feature_indices:
		feature_index_array = np.loadtxt(args.feature_indices,dtype=int)
		empirical_features = empirical_features[:,feature_index_array]
		out_file_stem = '.'.join(os.path.basename(args.feature_indices).split('.')[:-1])
	else:
		out_file_stem = os.path.basename(model_name)
	out_file_stem = out_file_stem + args.outname
	# scale data using the min-max scaler (between 0 and 1)

	if file_training_labels:
		size_output = len(set(training_labels))
	elif args.n_labels:
		# get the number of labels
		size_output = args.n_labels
	elif args.outlabels:
		size_output = len(args.outlabels)
	else:
		quit('Missing value ERROR: Use the "-n_labels" flag to specify the number of categories or provide the label array used for training using the "-l" flag or provide "-outlabels" flag followed by a list of desired label names.')

	print("Loading weights...")
	# DEF SIZE OF THE FEATURES
	hSize = empirical_features.shape[1]
	model=Sequential() # init neural network
	model.add(Dense(input_shape=(hSize,),units=int(units_multiplier[0]*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
	for jj  in range(n_hidden_layers-1):
		model.add(Dense(units=int(units_multiplier[jj+1]*hSize),activation=activation_function,kernel_initializer=kernel_init,use_bias=True))
	model.add(Dense(units=size_output,activation="softmax",kernel_initializer=kernel_init,use_bias=True))
	model.summary()
	model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
	model.load_weights(model_name, by_name=False)
	print("done.")
	#print(model.get_config())
	print(np.shape(empirical_features))

	estimate_par = model.predict(empirical_features)
	print(estimate_par)
	outfile = os.path.join(outpath,"label_probabilities_%s.txt" %out_file_stem)
	np.savetxt(outfile, np.round(estimate_par,4), delimiter="\t",fmt='%1.4f')
	

	if file_training_labels:
		try:
			lab = np.sort(np.arange(list(set(training_labels))).astype(int))
		except:
			lab = np.sort(np.array(list(set(training_labels))))
	elif args.outlabels:
		lab = np.array(args.outlabels)
	else:
		lab=np.arange(size_output)
	indx_best = np.argmax(estimate_par,axis=1)
   
	outfile = os.path.join(outpath,"labels_%s.txt" %out_file_stem)
	np.savetxt(outfile, lab[indx_best], delimiter="\t",fmt="%s")
