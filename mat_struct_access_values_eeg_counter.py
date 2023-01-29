import scipy.io as sio
import numpy
from numpy import savetxt 
from sklearn.decomposition import PCA
from tensorflow.random import set_seed
from numpy.random import seed
from sklearn.preprocessing import RobustScaler

def _check_keys( dict):
	"""
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	"""
	for key in dict:
    		if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
        		dict[key] = _todict(dict[key])
	return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
	"""
	this function should be called instead of direct scipy.io .loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	"""
	data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	return _check_keys(data)

# setting the seed
seed(1)
set_seed(1)

# create the robust scaler for the data
rScaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(20, 100-20), unit_variance=True)

labelZero = numpy.zeros((100, 1))
epochs = numpy.empty((0, 64))
my_input_data = numpy.empty((0, 64))
input_data = numpy.empty((0, 64))
final_input_data = numpy.empty((0, 32000))
input_to_nn = numpy.empty((0, 32000))

for i in range (100):
	myKeys = loadmat("Nothing\\Tr_Rescale_EEG_Data_Nothing" + str(i+1) + ".mat")
	print(myKeys)
	eegData = myKeys['EEG_Data']
	for k in range (400, 1400, 2):
		epochs = eegData[k]
		my_input_data = numpy.append(my_input_data, epochs.reshape(1, 64), axis=0)
	print(my_input_data.shape)						

	input_data = rScaler.fit_transform(my_input_data)	
	input_data = input_data.transpose()
	input_to_nn = input_data.flatten().reshape(1, 32000)
	final_input_data = numpy.append(final_input_data, input_to_nn, axis=0)
	my_input_data = numpy.empty((0, 64))

final_input_data_with_labels_0 = numpy.append(final_input_data, labelZero, axis=1)

labelOne = numpy.ones((100, 1))
final_input_data = numpy.empty((0, 32000))

for i in range (100):
	myKeys = loadmat("Pull\\Tr_Rescale_EEG_Data_Pull" + str(i+1) + ".mat")
	print(myKeys)
	eegData = myKeys['EEG_Data']
	for k in range (400, 1400, 2):
		epochs = eegData[k]
		my_input_data = numpy.append(my_input_data, epochs.reshape(1, 64), axis=0)
	print(my_input_data.shape)						

	input_data = rScaler.fit_transform(my_input_data)	
	input_data = input_data.transpose()
	input_to_nn = input_data.flatten().reshape(1, 32000)
	final_input_data = numpy.append(final_input_data, input_to_nn, axis=0)
	my_input_data = numpy.empty((0, 64))

final_input_data_with_labels_1 = numpy.append(final_input_data, labelOne, axis=1)

labelTwo = numpy.ones((100, 1)) * 2
final_input_data = numpy.empty((0, 32000))

for i in range (100):
	myKeys = loadmat("Push\\Tr_Rescale_EEG_Data_Push" + str(i+1) + ".mat")
	print(myKeys)
	eegData = myKeys['EEG_Data']
	for k in range (400, 1400, 2):
		epochs = eegData[k]
		my_input_data = numpy.append(my_input_data, epochs.reshape(1, 64), axis=0)
	print(my_input_data.shape)						

	input_data = rScaler.fit_transform(my_input_data)	
	input_data = input_data.transpose()
	input_to_nn = input_data.flatten().reshape(1, 32000)
	final_input_data = numpy.append(final_input_data, input_to_nn, axis=0)
	my_input_data = numpy.empty((0, 64))

final_input_data_with_labels_2 = numpy.append(final_input_data, labelTwo, axis=1)

final_input_data_with_labels = numpy.concatenate((final_input_data_with_labels_0, final_input_data_with_labels_1, final_input_data_with_labels_2), axis=0)	

savetxt('combined_eeg_data_with_labels_22_11_11_final.csv', final_input_data_with_labels, delimiter=',')

transposed_data = numpy.transpose(final_input_data_with_labels)
savetxt('transposed_data.csv', transposed_data, delimiter=',')


