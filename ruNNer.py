# python3
# Created by Daniele Silvestro on 2019.05.23
import matplotlib
import datetime
matplotlib.use("Agg")
# import keras
import numpy as np
from numpy import *
import scipy.special

np.set_printoptions(suppress=1)  # prints floats, no scientific notation
np.set_printoptions(precision=3)  # rounds all array elements to 3rd digit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# from tensorflow import set_random_seed
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import tensorflow as tf

# import tensorflow.keras.backend
from tensorflow.keras import backend
import argparse, sys, copy
import os

try:
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable tf compilation warning
except:
	pass

p = argparse.ArgumentParser()  # description='<input file>')
p.add_argument("-mode", choices=["train", "predict", "test"], default=None, required=True)
p.add_argument("-t", type=str, help="array of training features", default="", metavar="")
p.add_argument("-l", type=str, help="array of training labels", default=0, metavar=0)
p.add_argument("-e", type=str, help="array of empirical features", default=0, metavar=0)
p.add_argument("-r", type=str, help="file with rescaling array or float", default=1, metavar=1)
p.add_argument("-feature_indices", type=str, help="array of feature indices to select", default=0, metavar=0, nargs = "+")
p.add_argument("-head", type=int, help="header in training or empirical features file", default=0, metavar=0)
p.add_argument("-train_instance_indices",type=str,help="array of indices for selecting training instances",default=0,metavar=0)
p.add_argument("-test",type=float,help="fraction of training used as test set",default=0.1,metavar=0.1)
p.add_argument("-outlabels", type=str, nargs="+", default=[])
p.add_argument("-layers", type=int, help="n. hidden layers", default=1, metavar=1)
p.add_argument("-dropout", type=float, default=[], metavar=[], nargs = "+")
p.add_argument("-outpath", type=str, help="", default="")
p.add_argument("-outname", type=str, help="", default="")
p.add_argument("-batch_size", type=int, help="if 0: dataset is not sliced into smaller batches", default=0, metavar=0)
p.add_argument("-epochs", type=int, help="", default=1000, metavar=1000)
p.add_argument("-rescale_data", type=int, help="If set to 0 data are not rescaled between 0 and 1; 2: standardization", default=1, metavar=1)
p.add_argument("-class_weight", type=int, help="0) uniform weights; 1) weight for imbalanced classes ", default=1, metavar=1)
p.add_argument("-sub_sample_classes", type=int, help="0) use all data; 1) use sub-sampling to balance classes ", default=0, metavar=0)
p.add_argument("-optim_epoch", type=int, help="0) min loss function; 1) max validation accuracy", default=0)
p.add_argument("-verbose", type=int, help="", default=0, metavar=0)
p.add_argument("-loadNN", type=str, help="", default="", metavar="")
p.add_argument("-seed", type=int, help="", default=0, metavar=0)
p.add_argument("-actfunc", type=int, help="1) relu; 2) tanh; 3) sigmoid", default=1, metavar=1)
p.add_argument("-kerninit", type=int, help="1) glorot_normal; 2) glorot_uniform", default=1, metavar=1)
p.add_argument("-nodes", type=float, help="n. nodes (if > 2: multiplier of n. features)", nargs="+", default=[1.0])
p.add_argument("-randomize_data", type=float, help="shuffle order data entries", default=1, metavar=1)
p.add_argument("-threads", type=int, help="n. of threads (0: system picks an appropriate number)", default=0, metavar=0)
p.add_argument("-cross_val", type=int, help="Set number of cross validations to run. Set to 0 to turn off.", default=0)
p.add_argument("-validation_off", action="store_true",help='No validation set will be used when training the model, training will run until number of epochs set with "-epochs" flag', default=False)
p.add_argument("-no_bias_node", action="store_true",help='Turn off the bias node in the NN.', default=False)
args = p.parse_args()

if args.no_bias_node:
    useBiasNode = False
else:
    useBiasNode = True
args = p.parse_args()


#set_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#set_optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
set_optimizer = "adam"

out_activation_func = "softmax"  # "sigmoid" #
loss_function = "categorical_crossentropy"  # "binary_crossentropy" #  #"kullback_leibler_divergence" #  "mean_squared_error" #
# with "sparse_categorical_crossentropy" no one-hot encoding is required
print_full_test_output = 0

# NN SETTINGS
n_hidden_layers = args.layers  # number of extra hidden layers
max_epochs = args.epochs
batch_size_fit = args.batch_size  # batch size
units_multiplier = args.nodes  # number of nodes per input
if n_hidden_layers != len(units_multiplier):
	units_multiplier = np.repeat(units_multiplier[0], n_hidden_layers)
	print("Using node multiplier:", units_multiplier)
plot_curves = 1
train_nn = 1
test_nn = 0

randomize_data = args.randomize_data
# run_test_accuracy = 0
# run_tests = 0
if args.mode == "train":
	run_train = 1
	run_empirical = 0
elif args.mode == "predict":
	run_train = 0
	run_empirical = 1
elif args.mode == "test":
	run_train = 0
	run_empirical = 0
	test_nn = 1

try:
	rescale_factors = float(args.r)
except (ValueError):
	rescale_factors = np.loadtxt(args.r)

# SET SEEDS
if args.seed == 0:
	rseed = np.random.randint(1000, 9999)
else:
	rseed = args.seed
np.random.seed(rseed)
np.random.seed(rseed)
tf.random.set_seed(rseed)

# n_threads = args.threads
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=n_threads, inter_op_parallelism_threads=n_threads)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# keras.backend.set_session(sess)


activation_functions = ["relu", "tanh", "sigmoid"]
activation_function = activation_functions[args.actfunc - 1]

kernel_initializers = ["glorot_normal", "glorot_uniform"]
kernel_init = kernel_initializers[args.kerninit - 1]

dropout = args.dropout
if len(args.dropout) == 0:
	dropout = np.zeros(n_hidden_layers)



outpath = args.outpath
if outpath == "":
	outpath = os.path.dirname(args.t)
	if outpath == "":
		outpath = os.path.dirname(args.e)
elif not os.path.exists(outpath):
	os.makedirs(outpath)

if args.loadNN == "":
	if args.cross_val > 1:
		model_out = os.path.join(outpath, "cv%s" % args.outname)
		if not os.path.exists(model_out):
			os.makedirs(model_out)
	else:
		model_out = outpath

	input_file_raw = os.path.basename(args.t)
	input_file = os.path.splitext(input_file_raw)[0]  # file name without extension

	model_name = os.path.join(model_out, "%s_NN%s" % (input_file, args.outname))
	# model_name = os.path.join(model_out,"trained_model_NN_%slayers%sepochs%sbatch%s%s_%s" % (n_hidden_layers,max_epochs,batch_size_fit,activation_function,kernel_init,rseed))
else:
	model_name = args.loadNN

# input files
file_training_data = args.t  # empirical data
file_empirical_data = args.e  # empirical data
file_training_labels = args.l  # training labels


# if labels are provided read them, because they will be used by training or prediction mode
if file_training_labels:
	try:
		training_labels = np.loadtxt(file_training_labels)  # load txt file
	except:
		training_labels = np.load(file_training_labels).astype(float)  # load npy file
   
	# the following is necessary because of the following line that tries to get the min value of the array
	try:
		training_labels = training_labels.astype(int)
	except:
		quit('Labels must be integers and not text.') # load npy file

	if np.min(training_labels) > 0:
		training_labels = training_labels - np.min(training_labels)

if file_training_data:
	try:
		training_features = np.loadtxt(file_training_data,skiprows=args.head)  # load txt file
	except:
		training_features = np.load(file_training_data)  # load npy file
	training_features /= rescale_factors

	# scale data using the min-max scaler (between 0 and 1)
	if args.rescale_data == 1:
		scaler = MinMaxScaler()
		scaler.fit(training_features)
		training_features = scaler.transform(training_features)
	if args.rescale_data == 2:
		training_features = (training_features - np.mean(training_features, axis=0)) / np.std(training_features, axis=0) 


# process train dataset
train_nn = 0
if run_train:
	# select features and instances, if files provided:
	if args.feature_indices:
		try:
			feature_index_array = np.loadtxt(args.feature_indices[0], dtype=int)
		except:
			feature_index_array = np.array([int(i) for i in args.feature_indices])
		
		training_features = training_features[:, feature_index_array]
		

	if args.train_instance_indices:
		instance_index_array = np.loadtxt(args.train_instance_indices, dtype=int)
		training_features = training_features[instance_index_array, :]
		training_labels = training_labels[instance_index_array]

	if args.sub_sample_classes:
		count_per_category = np.unique(training_labels, return_counts = True)
		min_n_instances = np.min(count_per_category[1]) #count_per_category[0][np.argmin(count_per_category[1])]
		print("count per category:", len(training_labels))
		subsampled_indx = []
		for label_class in count_per_category[0]:
			indx_class = np.where(training_labels==label_class)[0]
			subsampled_indx = subsampled_indx + list(np.random.choice(indx_class, min_n_instances, replace=False))
		training_features = training_features[subsampled_indx, :] + 0
		training_labels = training_labels[subsampled_indx] + 0

	dSize = np.shape(training_features)[0]
	if randomize_data:
		rnd_indx = np.random.choice(np.arange(dSize), dSize, replace=False)
		# shuffle data
		training_features = training_features[rnd_indx, :] + 0
		training_labels = training_labels[rnd_indx] + 0
		

	init_training_features = copy.deepcopy(training_features)
	init_training_labels = copy.deepcopy(training_labels)
	train_indx = range(int(training_features.shape[0] * (1 - args.test)))
	if args.test:
		# split into training and test set
		test_indx = range(int(training_features.shape[0] * (1 - args.test)), training_features.shape[0])
		input_test = training_features[test_indx, :]
		input_testLabels = training_labels[test_indx].astype(int)
		input_testLabelsPr = tf.keras.utils.to_categorical(input_testLabels)
		input_training = training_features[train_indx, :]
		input_trainLabels = training_labels[train_indx].astype(int)
		input_trainLabelsPr = tf.keras.utils.to_categorical(input_trainLabels)
		test_nn = 1
	else:
		input_training = training_features
		input_trainLabels = training_labels
		input_trainLabelsPr = tf.keras.utils.to_categorical(input_trainLabels)
		test_nn = 0
	if batch_size_fit == 0:
		batch_size_fit = int(input_training.shape[0])

	print("\nTraining data shape:", input_training.shape)

	train_nn = 1

	# DEF SIZE OF THE FEATURES
	hSize = np.shape(input_training)[1]
	nCat = np.shape(input_trainLabelsPr)[1]
	dSize = np.shape(input_training)[0]

	index = input_training.shape[0]
	if args.cross_val > 1:
		training_data = []
		training_labels = []
		validation_data_list = []
		skf = StratifiedKFold(n_splits=int(args.cross_val))
		for train, test in skf.split(input_training, input_trainLabelsPr[:, 0]):
			training_data.append(input_training[train, :])
			training_labels.append(input_trainLabelsPr[train, :])
			validation_features = input_training[test, :]
			validation_labels = input_trainLabelsPr[test, :]
			validation_data_list.append((validation_features, validation_labels))
	elif args.validation_off:
		training_data = [input_training]
		training_labels = [input_trainLabelsPr]
		validation_features = []
		validation_labels = []
		validation_data_list = [(validation_features, validation_labels)]
	else:
		index = int(input_training.shape[0] * 0.8)
		training_data = [input_training[:index, :]]
		training_labels = [input_trainLabelsPr[:index, :]]
		validation_features = input_training[index:, :]
		validation_labels = input_trainLabelsPr[index:, :]
		validation_data_list = [(validation_features, validation_labels)]

	# GET CLASS WEIGHT
	if args.class_weight:
		# res = dict(zip(test_keys, test_values))
		from sklearn.utils import class_weight

		class_weights = class_weight.compute_class_weight(
			"balanced", np.unique(input_trainLabels[:index]), input_trainLabels[:index]
		)
		print("Estimated class weights:", class_weights)
	else:
		class_weights = np.ones(nCat)
		print("Using equal class weights:", class_weights)

	accuracy_scores = []
	loss_scores = []
	best_epochs = []
	if units_multiplier[0] < 2:
		multiplier_nodes = hSize
	else:
		multiplier_nodes = 1
	for i, input_training in enumerate(training_data):
		input_trainLabelsPr = training_labels[i]
		validation_data = validation_data_list[i]
		modelFirstRun = Sequential()  # init neural network
		
		### DEFINE from INPUT HIDDEN LAYER
		modelFirstRun.add(
			Dense(
				input_shape=(hSize,),
				units=int(units_multiplier[0] * multiplier_nodes),
				activation=activation_function,
				kernel_initializer=kernel_init,
				use_bias=useBiasNode,
			)
		)
		if dropout[0] > 0:
			modelFirstRun.add(
				Dropout( rate=dropout[0]				
				)
			)
		
		### ADD HIDDEN LAYER
		for jj in range(n_hidden_layers - 1):
			modelFirstRun.add(
				Dense(
					units=int(units_multiplier[jj + 1] * multiplier_nodes),
					activation=activation_function,
					kernel_initializer=kernel_init,
					use_bias=useBiasNode,
				)
			)
			
			if dropout[jj+1] > 0:
				modelFirstRun.add(
					Dropout( rate=dropout[jj]
					)
				)
			

		modelFirstRun.add(
			Dense(
				units=nCat,
				activation=out_activation_func,
				kernel_initializer=kernel_init,
				use_bias=False,
			)
		)
		modelFirstRun.summary()
		modelFirstRun.compile(loss=loss_function, optimizer="adam", metrics=["accuracy"])
		
		print("Running model.fit")
		log_dir= os.path.join(model_name, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
		
		# if no validation data (set by user) just train until final epoch
		if len(validation_data[0]) == 0:
			np.random.seed(rseed)
			np.random.seed(rseed)
			tf.random.set_seed(rseed)
			history = modelFirstRun.fit(
				input_training,
				input_trainLabelsPr,
				epochs=max_epochs,
				batch_size=batch_size_fit,
				verbose=args.verbose,
				class_weight=class_weights,
			)
			model = modelFirstRun
			print("Running training without validation set")
		else:
			np.random.seed(rseed)
			np.random.seed(rseed)
			tf.random.set_seed(rseed)
			import datetime
			history = modelFirstRun.fit(
				input_training,
				input_trainLabelsPr,
				epochs=max_epochs,
				batch_size=batch_size_fit,
				validation_data=validation_data,
				verbose=args.verbose,
				class_weight=class_weights,
				callbacks=[tensorboard_callback]
			)

			if plot_curves:
				fig = plt.figure(figsize=(20, 8))
				fig.add_subplot(121)
				plt.plot(history.history["loss"], "r", linewidth=3.0)
				plt.plot(history.history["val_loss"], "b", linewidth=3.0)
				plt.legend(["Training loss", "Validation Loss"], fontsize=12)
				plt.xlabel("Epochs", fontsize=12)
				plt.ylabel("Loss", fontsize=12)
				plt.title("Loss Curves", fontsize=12)

				# Accuracy Curves
				fig.add_subplot(122)
				# print(history.history)
				plt.plot(history.history["accuracy"], "r", linewidth=3.0)
				plt.plot(history.history["val_accuracy"], "b", linewidth=3.0)
				plt.legend(["Training Accuracy", "Validation Accuracy"], fontsize=12)
				plt.xlabel("Epochs", fontsize=12)
				plt.ylabel("Accuracy", fontsize=12)
				plt.title("Accuracy Curves", fontsize=12)

			# OPTIM OVER VALIDATION AND THEN TEST ON TEST DATASET (THAT'S THE FINAL ACCURACY)
			if args.optim_epoch == 0:
				optimal_number_of_epochs = np.argmin(history.history["val_loss"])
			elif args.optim_epoch == 1:
				optimal_number_of_epochs = np.argmax(history.history["val_accuracy"])
			best_epochs.append(optimal_number_of_epochs + 1)
			# print loss and accuracy at best epoch to file
			loss_at_best_epoch = history.history["val_loss"][optimal_number_of_epochs]
			accuracy_at_best_epoch = history.history["val_accuracy"][optimal_number_of_epochs]
			print("optimal number of epochs:", optimal_number_of_epochs+1, accuracy_at_best_epoch)			
			
			tnsordboard = "tensorboard --logdir %s " % log_dir
			print("\n\n To visualize the ouput type: \n%s\n\n" % tnsordboard)
			
			model = Sequential()  # init neural network
			model.add(
				Dense(
					input_shape=(hSize,),
					units=int(units_multiplier[0] * multiplier_nodes),
					activation=activation_function,
					kernel_initializer=kernel_init,
					use_bias=useBiasNode,
				)
			)
			
			if dropout[0] > 0:
				model.add(
					Dropout( rate=dropout[0]				
					)
				)
			
			for jj in range(n_hidden_layers - 1):
				model.add(
					Dense(
						units=int(units_multiplier[jj + 1] * multiplier_nodes),
						activation=activation_function,
						kernel_initializer=kernel_init,
						use_bias=useBiasNode,
					)
				)
				if dropout[jj+1] > 0:
					model.add(
						Dropout( rate=dropout[jj]
						)
					)
				
			model.add(
				Dense(
					units=nCat,
					activation=out_activation_func,
					kernel_initializer=kernel_init,
					use_bias=False,
				)
			)
			model.summary()
			model.compile(loss=loss_function, optimizer=set_optimizer, metrics=["accuracy"])
			np.random.seed(rseed)
			np.random.seed(rseed)
			tf.random.set_seed(rseed)
			history = model.fit(
				input_training,
				input_trainLabelsPr,
				epochs=optimal_number_of_epochs + 1,
				batch_size=batch_size_fit,
				validation_data=validation_data,
				verbose=args.verbose,
				class_weight=class_weights,
			)

			accuracy = history.history["val_accuracy"][-1]
			print("retrained valiadation accuracy:",accuracy)
			accuracy_scores.append(np.round(accuracy, 6))
			loss_scores.append(history.history["val_loss"][-1])

		if args.cross_val > 1:
			weight_file_name = model_name + "_cv_%i" % i
		else:
			weight_file_name = model_name

		weight_file_name = weight_file_name + ".model"
		# model.save_weights(weight_file_name)
		model.save(weight_file_name)
		print("Model saved as:", weight_file_name)

	try:
		# plot all accuracy curves (multiple pages in case of cv)
		file_name = "%s_res.pdf" % (model_name.replace("trained_model_", ""))
		pdf = matplotlib.backends.backend_pdf.PdfPages(file_name)
		for figure in range(1, fig.number + 1):
			pdf.savefig(figure)
		pdf.close()
		plt.close("all")
	except:
		no_plot = True

	# write output text file
	info_out = os.path.join( "%s_info.txt" % model_name.replace("trained_model_", "")
	)
	args_data = vars(args)
	# adjust seed since it may have been randomely drawn
	args_data["seed"] = rseed
	# add the shape of the training data input
	args_data["total_training_array_shape"] = str(input_training.shape)


	if not args.validation_off:
		cv_avg_epochs = int(np.round(np.mean(best_epochs)))
		cv_avg_accuracy = np.mean(accuracy_scores)
		cv_avg_loss = np.mean(loss_scores)
		args_data["best_epoch"] = str(best_epochs)
		args_data["accuracies"] = str((accuracy_scores))
		args_data["loss_scores"] = str((loss_scores))
		args_data["avg_epoch"] = str(cv_avg_epochs)
		args_data["avg_accuracy"] = str(cv_avg_accuracy)
		args_data["avg_loss"] = str(cv_avg_loss)
		print("Best epoch (average):", cv_avg_epochs)
		print("Validation accuracy (average):", cv_avg_accuracy)
		print(accuracy_scores)

	out_file = open(info_out, "w")
	for i in args_data:
		out_file.writelines(f"{i}\t{str(args_data[i])}\n")

if args.cross_val > 1:
	# re-train the NN based on entire dataset (no validation)
	# and run it on test set
	input_training = init_training_features[train_indx, :]
	input_trainLabels = init_training_labels[train_indx].astype(int)
	input_trainLabelsPr = tf.keras.utils.to_categorical(input_trainLabels)
	
	if units_multiplier[0] < 2:
		multiplier_nodes = hSize
	else:
		multiplier_nodes = 1
	
	model = Sequential()  # init neural network
	model.add(
		Dense(
			input_shape=(hSize,),
			units=int(units_multiplier[0] * multiplier_nodes),
			activation=activation_function,
			kernel_initializer=kernel_init,
			use_bias=useBiasNode,
		)
	)
	if dropout[0] > 0:
		model.add(
			Dropout( rate=dropout[0]				
			)
		)
	
	for jj in range(n_hidden_layers - 1):
		model.add(
			Dense(
				units=int(units_multiplier[jj + 1] * multiplier_nodes),
				activation=activation_function,
				kernel_initializer=kernel_init,
				use_bias=useBiasNode,
			)
			
		)
		
		if dropout[jj+1] > 0:
			model.add(
				Dropout( rate=dropout[jj]
				)
			)
		
	model.add(
		Dense(
			units=nCat,
			activation=out_activation_func,
			kernel_initializer=kernel_init,
			use_bias=False,
		)
	)
	model.summary()
	model.compile(loss=loss_function, optimizer=set_optimizer, metrics=["accuracy"])
	
	np.random.seed(rseed)
	np.random.seed(rseed)
	tf.random.set_seed(rseed)
	history = model.fit(
		input_training,
		input_trainLabelsPr,
		epochs=cv_avg_epochs,
		batch_size=batch_size_fit,
		verbose=args.verbose,
		class_weight=class_weights,
	)
	model.save(model_name + "_CV")
	# model.save_weights(model_name+"_CV")
	print("Model saved as:", model_name + ".CVmodel")


if test_nn and args.test > 0.0:  # and not args.cross_val > 1:
	if args.mode == "test":
		input_test = training_features
		if args.feature_indices:
			try:
				feature_index_array = np.loadtxt(args.feature_indices[0], dtype=int)
			except:
				feature_index_array = np.array([int(i) for i in args.feature_indices])
				input_test = input_test[:, feature_index_array]		
		input_testLabels = training_labels.astype(int)
		input_testLabelsPr = tf.keras.utils.to_categorical(input_testLabels)
		hSize = np.shape(input_test)[1]
		nCat = np.shape(input_testLabelsPr)[1]
		dSize = np.shape(input_test)[0]
		model = tf.keras.models.load_model(model_name)
		info_out = os.path.join(outpath, "%s_info.txt" % model_name.replace("_NN_", ""))
		out_file = open(info_out, "w")
		print(
			"\n\nUsing %.3f of the data as test set.\nDimensions of resulting test set: %s."
			% (args.test, str(input_test.shape))
		)

	print("\nTest data shape:", input_test.shape)
	estimate_par = model.predict(input_test)
	predictions = np.argmax(estimate_par, axis=1)
	if print_full_test_output:
		for i in range(len(estimate_par)):
			print(input_testLabels[i], estimate_par[i], input_testLabelsPr[i])

	cM = confusion_matrix(input_testLabels, predictions)
	print("Confusion matrix (test set):\n", cM)
	rescaled_cM = (np.array(cM).T / np.sum(np.array(cM), 1)).T
	print(rescaled_cM)
	scores = model.evaluate(input_test, input_testLabelsPr, verbose=0)
	print("\nTest accuracy rate: %.2f%%" % (scores[1] * 100))
	print("Test error rate: %.2f%%" % (100 - scores[1] * 100))
	print("Test cross-entropy loss:", round(scores[0], 3), "\n")
	out_file.writelines(f"\nTest accuracy rate\t%s " % (scores[1]))
	out_file.writelines(f"\nTest cross-entropy loss\t%s " % (scores[0]))
	out_file.writelines(f"\nConfusion matrix:\n%s" % cM)
	out_file.writelines(f"\nRescaled confusion matrix:\n%s" % rescaled_cM)

	if args.mode == "test":
		out_file = open(info_out.replace(".txt", "") + "_class_prob.txt", "w")
		out = "label\t"
		for i in np.unique(input_testLabels):
			out += "P_%s\t" % i
		out_file.writelines(out)
		for i in range(len(estimate_par)):
			line_list = [input_testLabels[i]] + list(estimate_par[i])
			out = "\n"
			for j in line_list:
				out = out + "%s\t" % j
			out_file.writelines(out)

		np.savetxt(
			info_out.replace(".txt", "") + "_CM.txt",
			rescaled_cM,
			fmt="%.3f",
			delimiter="\t",
		)


######## TEST EMPIRICAL DATA SETS
if run_empirical:
	print("Loading input file...")
	try:
		empirical_features = np.loadtxt(file_empirical_data,skiprows=args.head)
		# print(empirical_features.shape)
	except:
		empirical_features = np.load(file_empirical_data)

	empirical_features /= rescale_factors
	# print(np.amin(empirical_features, 0))
	# print(np.amax(empirical_features, 0))
	# select features and instances, if files provided:
	if args.feature_indices:
		try:
			feature_index_array = np.loadtxt(args.feature_indices[0], dtype=int)
		except:
			feature_index_array = np.array([int(i) for i in args.feature_indices])
		empirical_features = empirical_features[:, feature_index_array]
	
	out_file_stem = os.path.basename(model_name)
	input_file_stem = os.path.basename(file_empirical_data)
	input_file_stem = input_file_stem.replace(".txt","")
	input_file_stem = input_file_stem.replace(".npy","")
	
	
	out_file_stem = out_file_stem + args.outname

	# scale data using the min-max scaler (between 0 and 1)
	if args.rescale_data == 1:
		scaler = MinMaxScaler()
		scaler.fit(empirical_features)
		empirical_features = scaler.transform(empirical_features)
	if args.rescale_data == 2:
		empirical_features = (empirical_features - np.mean(empirical_features, axis=0)) / np.std(empirical_features, axis=0) 
	

	print("Loading model...")
	model = tf.keras.models.load_model(model_name)
	estimate_par = model.predict(empirical_features)
	# print(estimate_par.shape)
	size_output = estimate_par.shape[1]
	outfile = os.path.join(outpath, "%slabelsPr_%s.txt" % (input_file_stem, out_file_stem))
	
	if file_training_labels:
		try:
			lab = np.sort(np.arange(list(set(training_labels))).astype(int))
		except:
			lab = np.sort(np.array(list(set(training_labels))))
	elif args.outlabels:
		lab = np.array(args.outlabels)
	else:
		lab = np.arange(size_output)
	
	col_names = ""
	for i in lab: 
		col_names = col_names + "%s\t" % i
	np.savetxt(outfile, np.round(estimate_par, 4), delimiter="\t", fmt="%1.4f", header=col_names)
	print("\nResults saved as:", outfile, "\n")

	indx_best = np.argmax(estimate_par, axis=1)

	outfile = os.path.join(outpath, "%slabels_%s.txt" % (input_file_stem, out_file_stem))
	np.savetxt(outfile, lab[indx_best], delimiter="\t", fmt="%s")

try:
	out_file.close()
except:
	pass