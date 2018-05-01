import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
import numpy as np

x_train = np.load("x_train_out.npy")
y_train = np.load("y_train_out.npy")
x_test = np.load("x_test_out.npy")
y_test = np.load("y_test_out.npy")
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#z-score
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

num_classes = 10
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

baseMapNum = 32
weight_decay = 1e-4

model = Sequential()
model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
        optimizer="Adam",
        metrics=['accuracy'])

batch_size = 64
data_augmentation = True

if data_augmentation :
    print("Using Data augmentation")
    datagen = ImageDataGenerator(
        featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,  
        width_shift_range=0.1, height_shift_range=0.1, 
        horizontal_flip=True, vertical_flip=False)  
    
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=20,verbose=1,validation_data=(x_test,y_test))
    
else :
    print("Not Using Data augmentation")
    model.fit(x_train, y_train,
              batch_size=batch_size*4,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
model.save_weights("simple_cnn_adam_20.h5")
model.load_weights("simple_cnn_adam_20.h5")

if data_augmentation :
    print("Using Data augmentation")
    datagen = ImageDataGenerator(
        featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,  
        width_shift_range=0.1, height_shift_range=0.1,  
        horizontal_flip=True, vertical_flip=False)  
    
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=20,verbose=1,validation_data=(x_test,y_test))
    
else :
    print("Not Using Data augmentation")
    model.fit(x_train, y_train,
              batch_size=batch_size*4, epochs=epochs, validation_data=(x_test, y_test),
              shuffle=True)

model.save_weights("cifar_simple_cnn_adam_40.h5")
model.load_weights("cifar10_90.62.h5")