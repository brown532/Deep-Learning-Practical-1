from alexnet import *



(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images/255
test_images = test_images/255


train_epochs = 10
####################

'''
	Description:
	Baseline : relu, no normalization, no dropout, adam opt.

	Exp. 1 With drop out

	Exp 2: With drop out & weight decay-> l2(0.0005)

	Exp 3: with drop out & weight decay  & batch normalization
	
	Exp 4: with drop out & weight decay  & batch normalization activation: leaky relu

	Exp 5: with drop out & weight decay & batch normalization activation: sigmoid

	Exp 6: with drop out & weight decay & batch normalization activation: relu, optimizer: sgd

	Exp 7: with drop out & weight decay & batch normalization activation: relu, optimizer: sgd with momentum 
'''


#Model 0: Baseline: relu
print("\n\nModel 0:")
alex_model = create_alex_model(activation='relu')
# #train model on training set
alex_model.fit(train_images, train_labels, batch_size=32, epochs=train_epochs,verbose=1)
test_loss, test_acc = alex_model.evaluate(test_images, test_labels,verbose=2)

print("Model 0: \tTest loss: "+str(test_loss)+"\tTest accuracy: "+str(test_acc)+"\n\n")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")



#Model 1: with drop out
alex_model = create_alex_model(drop_out=True,activation='relu')
print("\n\nModel 1:")
# #train model on training set
alex_model.fit(train_images, train_labels, batch_size=32, epochs=train_epochs,verbose=1)
test_loss, test_acc = alex_model.evaluate(test_images, test_labels,verbose=2)
print("Model 1: \tTest loss: "+str(test_loss)+"\tTest accuracy: "+str(test_acc)+"\n\n")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")



#Model 2: with drop out and weight decay
alex_model = create_alex_model(drop_out=True,weight_decay=True,activation='relu')
print("\n\nModel 2:")
# #train model on training set
alex_model.fit(train_images, train_labels, batch_size=32, epochs=train_epochs,verbose=1)
test_loss, test_acc = alex_model.evaluate(test_images, test_labels,verbose=2)
print("Model 2: \tTest loss: "+str(test_loss)+"\tTest accuracy: "+str(test_acc)+"\n\n")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

#Model 3: with drop out, weight decay and batch normalization
alex_model = create_alex_model(drop_out=True,weight_decay=True, batch_normalization=True, activation='relu')
print("\n\nModel 3:")
# #train model on training set
alex_model.fit(train_images, train_labels, batch_size=32, epochs=train_epochs,verbose=1)
test_loss, test_acc = alex_model.evaluate(test_images, test_labels,verbose=2)
print("Model 3: \tTest loss: "+str(test_loss)+"\tTest accuracy: "+str(test_acc)+"\n\n")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")



#Model 4: with drop out,weight decay, leaky relu
alex_model = create_alex_model(drop_out=True,weight_decay=True, batch_normalization=True,activation='lrelu')
print("\n\nModel 4:")
# #train model on training set
alex_model.fit(train_images, train_labels, batch_size=32, epochs=train_epochs,verbose=1)
test_loss, test_acc = alex_model.evaluate(test_images, test_labels,verbose=2)
print("Model 4: \tTest loss: "+str(test_loss)+"\tTest accuracy: "+str(test_acc)+"\n\n")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


#Model 5: with drop out,weight decay, sigmoid
alex_model = create_alex_model(drop_out=True,weight_decay=True, batch_normalization=True,activation='sigmoid')
print("\n\nModel 5:")
# #train model on training set
alex_model.fit(train_images, train_labels, batch_size=32, epochs=train_epochs,verbose=1)
test_loss, test_acc = alex_model.evaluate(test_images, test_labels,verbose=2)
print("Model 5: \tTest loss: "+str(test_loss)+"\tTest accuracy: "+str(test_acc)+"\n\n")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


#Model 6: with drop out,weight decay, relu, SGD
alex_model = create_alex_model(drop_out=True,weight_decay=True, batch_normalization=True,activation='relu',optimizer='sgd')
print("\n\nModel 6:")
# #train model on training set
alex_model.fit(train_images, train_labels, batch_size=32, epochs=train_epochs,verbose=1)
test_loss, test_acc = alex_model.evaluate(test_images, test_labels,verbose=2)
print("Model 6: \tTest loss: "+str(test_loss)+"\tTest accuracy: "+str(test_acc)+"\n\n")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


#Model 7: with drop out,weight decay, relu, SGD with momentum
alex_model = create_alex_model(drop_out=True,weight_decay=True, batch_normalization=True, activation='relu',optimizer='sgd with momentum')
print("\n\nModel 7:")
# #train model on training set
alex_model.fit(train_images, train_labels, batch_size=32, epochs=train_epochs,verbose=1)
test_loss, test_acc = alex_model.evaluate(test_images, test_labels,verbose=2)
print("Model 7: \tTest loss: "+str(test_loss)+"\tTest accuracy: "+str(test_acc)+"\n\n")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

