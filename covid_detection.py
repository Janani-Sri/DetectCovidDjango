# importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

print("0---------")

#initializing the CNN
classifier = Sequential()
print("1---------")
#convolution step 1
classifier.add(Convolution2D(96,11,strides=(4,4),padding='valid',input_shape=(224,224,3),activation='relu'))
print("2---------")
#max pooling step 1
classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
classifier.add(BatchNormalization())
print("3---------")
#convolution step 2
classifier.add(Convolution2D(256,11,strides=(1,1), padding='valid', activation='relu'))
print("4---------")
#max pooling step 2
classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
classifier.add(BatchNormalization())
print("5---------")
#convolution step 3
classifier.add(Convolution2D(384,3,strides=(1,1),padding='valid',activation='relu'))
classifier.add(BatchNormalization())
print("6---------")
#convolution step 4
classifier.add(Convolution2D(384,3,strides=(1,1),padding='valid',activation='relu'))
classifier.add(BatchNormalization())
print("7---------")
#convolution step 5
classifier.add(Convolution2D(256,3,strides=(1,1),padding='valid',activation='relu'))
print("8---------")
#max pooling step 3
classifier.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
classifier.add(BatchNormalization())
print("9---------")
#flattening step
classifier.add(Flatten())
print("10---------")
#full connection step
classifier.add(Dense(units=4096, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units=4096, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units=1000,activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(Dense(units=3,activation='softmax'))
classifier.summary()

print("11---------")
#compiling the CNN
classifier.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
print("12---------")
#image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rotation_range=40,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
print("13---------")
test_datagen = ImageDataGenerator(rescale=1./255)
print("14---------")
batch_size = 32

train_data_dir = "D:/Academics/FINAL YEAR PROJECT/dataset/train"
print("15---------")
test_data_dir = "D:/Academics/FINAL YEAR PROJECT/dataset/test"
print("16---------")
training_set = train_datagen.flow_from_directory(train_data_dir,
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')
print("17---------")
test_set = test_datagen.flow_from_directory(test_data_dir,
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical')
print("18---------")
print(training_set.class_indices)
print("19---------")

#fitting images to CNN
history = classifier.fit_generator(training_set,
                                   steps_per_epoch=training_set.samples//batch_size,
                                   validation_data=test_set,
                                   epochs=50,
                                   validation_steps=test_set.samples//batch_size)
print("20---------")
#saving model
filepath = "classify/model.hdf5"
classifier.save(filepath)
print("21---------")
# plotting training values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
print("22---------")
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
print("23---------")
#accuracy plot
plt.plot(epochs,acc,color='green',label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
print("24---------")
#loss plot
plt.plot(epochs,loss, color='pink',label= 'Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
print("25---------")
plt.show()
print("26---------")