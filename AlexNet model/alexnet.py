import keras
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.activations import  relu

def create_alex_model(drop_out=False,batch_normalization=False,weight_decay=False,activation='relu',optimizer='adam'):
  
	if weight_decay == True:
		reg = l2(0.0005)
	else:
		reg = None
	

	if activation == 'relu':
		activation = 'relu'
	elif activation == 'lrelu':
		activation = lambda x: keras.activations.relu(x, alpha=0.1)
	elif activation == 'sigmoid':
		activation ='sigmoid'

	if optimizer == 'adam':
		optimizer = 'adam'
	elif optimizer == 'sgd':
		optimizer = SGD(lr=0.01)
	elif optimizer == 'sgd with momentum':
		optimizer = SGD(lr=0.01, momentum=0.9)


	#create model
	model = Sequential()

	model.add(Conv2D(32, (11, 11), strides = 1, padding = "same", kernel_regularizer=reg, activation = activation, input_shape = (32, 32, 3)))
	if batch_normalization ==True:
		model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))


	model.add(Conv2D(32, (5, 5), strides = 1, padding = "same", activation = activation, kernel_regularizer=reg))

	if batch_normalization == True:
		model.add(BatchNormalization())

	model.add(MaxPooling2D((2, 2)))


	model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = activation, kernel_regularizer=reg))
	if batch_normalization == True:
		model.add(BatchNormalization())

	model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = activation, kernel_regularizer=reg))
	if batch_normalization == True:
		model.add(BatchNormalization())
	


	model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = activation, kernel_regularizer=reg))
	if batch_normalization == True:
		model.add(BatchNormalization())

	model.add(MaxPooling2D((2, 2)))

	##############################################

	model.add(Flatten())

	model.add(Dense(512, kernel_regularizer=reg))

	if batch_normalization == True:
		model.add(BatchNormalization())

	if drop_out==True:
		model.add(Dropout(0.4))

	model.add(Dense(512, kernel_regularizer=reg))

	if batch_normalization == True:
		model.add(BatchNormalization())

	if drop_out==True:
		model.add(Dropout(0.4))

	model.add(Dense(1024, kernel_regularizer=reg))

	if batch_normalization == True:
		model.add(BatchNormalization())

	if drop_out==True:
		model.add(Dropout(0.4))
	model.add(Dense(10, activation = 'softmax', kernel_regularizer=reg))


	model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

	return model