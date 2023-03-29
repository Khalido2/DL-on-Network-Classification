import numpy as np
import math
import datasetBuilder3
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.models import Sequential

#Retreive training and testing set
NUMBER_OF_SAMPLES = 15065 #number of PIMs in each class
TRAINING_SIZE = math.ceil(NUMBER_OF_SAMPLES *0.7)
TEST_SIZE = math.floor(NUMBER_OF_SAMPLES *0.2)
VAL_SIZE = math.floor(NUMBER_OF_SAMPLES *0.1)

sampleSize = 6
paddingSize = 12

#This function creates a model based on the given parameters and trains it on the data
def get_keras_model(numFilters, firstFilterSize, secondFilterSize,numNeuronsDense, numDense,numNeuronsDenseLast, dropout, activationConv, activationDense):

    inputShape = (12,12,1)
    model = Sequential()
    model.add(Conv2D(numFilters, kernel_size=firstFilterSize, strides=1, activation=activationConv, input_shape=inputShape, padding='same'))  # C1
    model.add(MaxPooling2D(pool_size=(2, 2)))  # S2
    model.add(Conv2D(numFilters*2, kernel_size=secondFilterSize, strides=1, activation=activationConv, input_shape=inputShape, padding='same'))  # C3

    model.add(MaxPooling2D(pool_size=(2, 2)))#S4

    model.add(Flatten())  # Flatten
    for i in range(numDense):
        model.add(Dense(numNeuronsDense, activation=activationDense))
        model.add(Dropout(dropout))

    model.add(Dense(numNeuronsDenseLast, activation=activationDense))
    model.add(Dropout(dropout))
    model.add(Dense(13, activation='softmax'))

    return model

# This function takes in the hyperparameters and returns the optimisation loss score of the trial.
def keras_mlp_cv_score(parameterization, weight=None):

    trainSet, trainLabels = datasetBuilder3.build2DFeatureSet(sampleSize, "trainDataset103.csv", paddingSize,TRAINING_SIZE) #get training and validation sets with labels
    valSet, valLabels = datasetBuilder3.build2DFeatureSet(sampleSize, "valDataset103.csv", paddingSize, VAL_SIZE)
    trainSet = trainSet.reshape(trainSet.shape[0], paddingSize, paddingSize, 1)
    valSet = valSet.reshape(valLabels.shape[0], paddingSize, paddingSize, 1)

    model = get_keras_model(parameterization.get('numFilters'),
                            parameterization.get('firstFilterSize'),
                            parameterization.get('secondFilterSize'),
                            parameterization.get('numNeuronsDense'),
                            parameterization.get('numDense'),
                            parameterization.get('numNeuronsDenseLast'),
                            parameterization.get('dropout'),
                            parameterization.get('convolution_activation'),
                            parameterization.get('dense_activation'))


    numEpochs = parameterization.get('epochs')
    learningRate = parameterization.get('learningRate')
    batchSize = parameterization.get('batchSize')

    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)  # stop training early if loss doesn't improve in 5 epochs

    optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    res = model.fit(x=trainSet, y=trainLabels, validation_data= (valSet, valLabels), batch_size=batchSize,epochs=numEpochs , callbacks=[callback])

    lastAccuracy = res.history['val_accuracy'][-1] #get final validation accuracy of model after training

    return 1 - lastAccuracy #return optimisation loss value

#The hyperparameters an their value range
parameters = [
    {
        "name": "learningRate",
        "type": "range",
        "bounds": [0.0001, 0.01],
        "log_scale": True,
        "value_type": "float"
    },
    {
        "name": "dropout",
        "type": "range",
        "bounds": [0, 0.2],
        "value_type": "float"
    },
    {
        "name": "numFilters",
        "type": "range",
        "bounds": [10, 100],
        "value_type": "int"
    },
    {
        "name": "firstFilterSize",
        "type": "choice",
        "values": [3, 5, 7],
        "value_type": "int"
    },
    {
        "name": "secondFilterSize",
        "type": "choice",
        "values": [3, 5],
        "value_type": "int"
    },
    {
        "name": "numDense",
        "type": "range",
        "bounds": [2, 4],
        "value_type": "int"
    },
    {
        "name": "numNeuronsDense",
        "type": "range",
        "bounds": [13, 256],
        "value_type": "int"
    },
    {
        "name": "numNeuronsDenseLast",
        "type": "range",
        "bounds": [13, 128],
        "value_type": "int"
    },
    {
        "name": "batchSize",
        "type": "choice",
        "values": [64, 128, 256, 512],
    },

    {
        "name": "epochs",
        "type": "choice",
        "values": [20, 40, 60],
    },

    {
        "name": "convolution_activation",
        "type": "choice",
        "values": ['tanh', 'relu'],
    },
    {
        "name": "dense_activation",
        "type": "choice",
        "values": ['sigmoid', 'relu'],
    }
]
