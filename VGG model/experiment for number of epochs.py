from vggmodel import *
from keras.callbacks import EarlyStopping
from matplotlib import pyplot



(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images/255
test_images = test_images/255


val_images = train_images[40000:50000]
val_labels = train_labels[40000:50000]

train_images = train_images[0:40000]
train_labels = train_labels[0:40000]


train_epochs = 300

print(train_images.shape)


VGG_model = create_vgg_model(activation='relu')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=30)

history = VGG_model.fit(train_images, train_labels, validation_data=(val_images, val_labels), batch_size=32, epochs=train_epochs, verbose=1, callbacks=[es])



_, train_acc = VGG_model.evaluate(train_images, train_labels, verbose=2)
_, val_acc = VGG_model.evaluate(val_images, val_labels, verbose=2)
_, test_acc = VGG_model.evaluate(test_images, test_labels, verbose=2)

print('Train: %.3f, Validation: %.3f Test: %.3f' % (train_acc, val_acc, test_acc))
# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.legend()
pyplot.show()


pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='validation')
pyplot.xlabel("epoch")
pyplot.ylabel("accuracy")
pyplot.legend()
pyplot.show()