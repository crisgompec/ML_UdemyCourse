# Convolutional Neural Network
"""
Our brain looks for **features** in a dataset. 


STEP 1A - CONVOLUTION
With the input image, we apply a filter over it, and we obtain then a feature
map. The feature map basically tell as where the highest coincidents between the
filter and input took place. Also, the resulting matrix has a lower dimension 
(is has been compressed using an intergral).
We create several feature maps, using different filters that are actually looking
for certain features. The filter are optimized along rounds in addition to the 
weights of the neural network.

STEP 1B - RELU LAYER
We want to increase non-linearity because images are non-linear. So we apply the
rely activation function to the results of the feature maps. 

STEP 2 - MAX POOLING
Based on spatial invariance: da igual el lugar de la imagen de las features, 
lo importante es que esta ahi la imagen (able to find that feature no matter conditions).
Large numbers in the feature indicate where filter matches the most with the 
image. Pooling consiste en quedarse con el maximo valor dentro de submatrices 
del feature map. We preserve the features and reduces the size of the feature map.

STEP 3 - FLATTENING
Basicamente consiste en poner las matrices obtenidas del pooling process en
un vector, we 'flat' them, which will be the input for an artificial neural
network.

Step 4 - FULL CONNECTION
We apply a neural network that will have many outputs as number of classes 
that we want to clasify. The the last connections to the output will learn
values for their weights such that the signals coming from previous neurons
are significant to certain clases.

During the backpropagation both the weights of the neural network and the 
filters of the convolution will change to optimize the detection of classes.

"""

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing

# Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Adding a second convolutional layer
# We can improve overfitting and accuracy of the model by increasing these hidden 
# layers!
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit_generator(training_set,
                  steps_per_epoch = 334,
                  epochs = 25,
                  validation_data = test_set,
                  validation_steps = 334)