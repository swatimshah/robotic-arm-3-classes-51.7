import numpy
from tensorflow.random import set_seed
from numpy.random import seed

# setting the seed
seed(1)
set_seed(1)

loaded_complete_data_1 = numpy.loadtxt('combined_eeg_data_with_labels_22_11_03_final.csv', delimiter=',')
loaded_complete_data_2 = numpy.loadtxt('combined_eeg_data_with_labels_22_11_11_final.csv', delimiter=',')
loaded_complete_data_3 = numpy.loadtxt('combined_eeg_data_with_labels_22_11_24_final.csv', delimiter=',')
loaded_complete_data_4 = numpy.loadtxt('combined_eeg_data_with_labels_22_11_28_final.csv', delimiter=',')
loaded_complete_data_5 = numpy.loadtxt('combined_eeg_data_with_labels_22_12_13_final.csv', delimiter=',')
loaded_complete_data_6 = numpy.loadtxt('combined_eeg_data_with_labels_22_12_15_final.csv', delimiter=',')

loaded_complete_data = numpy.concatenate((loaded_complete_data_1, loaded_complete_data_2, loaded_complete_data_3, loaded_complete_data_4, loaded_complete_data_5, loaded_complete_data_6), axis=0)	

numpy.savetxt('loaded_complete_data_2_sec.csv', loaded_complete_data, delimiter=',')	
