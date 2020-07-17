# importing libraries 
import os
import keras
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.callbacks import History
from keras.preprocessing import image 
from keras import backend as K 
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

#Set Variables
#set width and height for resizing the training data
img_width, img_height = 224, 224
num_classes = 5 #number of classes for this use case (Bauhaus, Expressionismus, Impressionismus, Romantik, Rennaisance)
train_data_dir = os.path.abspath("multi_epochen_data/train")
validation_data_dir = os.path.abspath("multi_epochen_data/test")
nb_train_samples = 3500 #number of train samples
nb_validation_samples = 500 #number of test samples
epochs = 30
batch_size = 64

# Makes sure the input data is recognized with the right shape, as different keras backends use different naming conventions
# (some like theano using "channels_first", tensorflow using "channels_last")
# https://www.codesofinterest.com/2017/09/keras-image-data-format.html
if K.image_data_format() == 'channels_first': 
	input_shape = (3, img_width, img_height) 
else: 
	input_shape = (img_width, img_height, 3) 


# Generate Data from pictures
# original images consist of RGB coefficients between 0-255. As this is to high for a typical model to process, all the values are rescaled to a value between 0 and 1
# define in what ways to alter the image (into new seperate images)

train_datagen = ImageDataGenerator( 
				rescale = 1. / 255, 
				shear_range = 0.2, 
				zoom_range = 0.2, 
                horizontal_flip = True,
                rotation_range=45,
                width_shift_range=0.2,
                height_shift_range=0.2,
                ) 
#just resizing, we do not want to alter the test data in other ways
test_datagen = ImageDataGenerator(rescale = 1. / 255) 

# Get all the images and process them
# train_datagen is of the ImageDataGenerator class
# the arguments it takes should be self explanatory, as the variables that get assigned have all been created above
# class_mode = 'categorical' tells the generator that we are working with more than 2 classes (as opposed to class_mode = 'binary')
train_generator = train_datagen.flow_from_directory(train_data_dir, 
							target_size =(img_width, img_height), 
					batch_size = batch_size, class_mode ='categorical') 

validation_generator = test_datagen.flow_from_directory( 
									validation_data_dir, 
				target_size =(img_width, img_height), 
		batch_size = batch_size, class_mode ='categorical') 


#create model structure
#the code commentary will not go into great detail here
#this model has been created through a lot of trial and error
#explanations to parts of this model, as needed, can be found in the project report
model = Sequential() 

model.add(Conv2D(64, (3, 3), activation='relu', input_shape = input_shape)) 
model.add(MaxPooling2D(pool_size =(2, 2))) #reduce size without loosing features
model.add(Dropout(0.2)) #reduce overfitting

model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
model.add(Dropout(0.2))

model.add(Flatten()) # make 1D array out of 2D image
model.add(Dense(128, activation='relu')) 
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.3))

model.add(Dense(num_classes))
#softmax is used to take the numerics of the last layer and turn them into probabilities, so the output vektor sums up to one
model.add(Activation('softmax'))

#an optimizer is chosen and given its parameters
rms = keras.optimizers.RMSprop(learning_rate = 0.001, rho=0.9)
# compile all the defined layers

model.compile(loss ='categorical_crossentropy', #categorical crossentropy, as we have more than 2 labels
					optimizer =rms, 
				metrics =['categorical_accuracy']) 


history = History()
#this is the step where the model that has been defined before is trained
model.fit_generator(train_generator,
	steps_per_epoch = nb_train_samples // batch_size, 
	epochs = epochs, validation_data = validation_generator, 
	validation_steps = nb_validation_samples // batch_size,
                   callbacks=[history])

model.save_weights('multi_epochen_weights.h5') 
model.save('multi_epochen.model')



# metrics
# the code below is used to graphically show different performance metrics of the CNN
# it is not relevant for the actual model

# plot training and validation accuracy per epoch
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot training and validation loss per epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    source: https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html (last access date: 16.07.2020)
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# load the model and it't weights. Make sure the correct path is selected!
model = tf.keras.models.load_model(os.path.abspath('multi_epochen.model'))
model.load_weights(os.path.abspath('multi_epochen_weights.h5'))

# create a generator composed of all validation data images
manual_test_datagen = ImageDataGenerator(rescale = 1. / 255) 
manual_test_generator= test_datagen.flow_from_directory( 
									validation_data_dir, 
				target_size =(img_width, img_height), 
		batch_size = 500, class_mode ='categorical') 

test_imgs, test_labels = next(manual_test_generator)

# predict on the items of the manual_test_generator
predictions = model.predict_generator(manual_test_generator, steps = 1, verbose = 2)

# convert one hot encoding into array of one integer that represents the class
predictions = tf.argmax(predictions, axis=1)
test_labels = tf.argmax(test_labels, axis=1)

# create confusion matrix and define its labels
cm = confusion_matrix(test_labels, predictions)
cm_plot_labels = ['Bauhaus', 'Expressionismus', 'Impressionismus', 'Renaissance', 'Romantik']   

# plot confusion matrix
plot_confusion_matrix(cm, cm_plot_labels, title="Confusion Matrix")