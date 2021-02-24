import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Dropout, ActivityRegularization
from keras.models import Sequential
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.activations import  relu

#create model

def standard_CNN_Model(drop_out=False,batch_normalization=False,weight_decay=False,activation='relu',optimizer='adam'):
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

  model = Sequential()
  #add model layers
  # model = Sequential()
  model.add(Conv2D(64, (3,3), activation=activation,padding='same', input_shape=(32,32,3)))
  if weight_decay == True:
    model.add(ActivityRegularization(l2=0.005))
  if batch_normalization == True:
    model.add(BatchNormalization())

  ##################################

  model.add(Conv2D(32, (3,3), padding='same', activation=activation))
  if weight_decay == True:
    model.add(ActivityRegularization(l2=0.005))
  if batch_normalization == True:
    model.add(BatchNormalization())

  ##############################
  model.add(Flatten())
  if drop_out == True:
    model.add(Dropout(0.4))

  model.add(Dense(10, activation='softmax'))
  model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

  return model