import math

import keras.callbacks
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.models import Sequential

import datasetBuilder3
import tensorflow as tf

NUMBER_OF_SAMPLES = 15065
TRAINING_SIZE = math.ceil(NUMBER_OF_SAMPLES *0.7)
VAL_SIZE = math.floor(NUMBER_OF_SAMPLES *0.1)
TEST_SIZE = math.floor(NUMBER_OF_SAMPLES *0.2)

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3) #stop training early if loss doesn't improve in 3 epochs

#Gets training set with sample size of k and returns validation accuracy of the model
def trainModel(sampleSize, model):
    results = []
    valLoss = 0
    valAccuracy = 0
    trainAccuracy = 0
    trainLoss = 0
    trainSet, trainLabels = datasetBuilder3.build2DFeatureSet(sampleSize, "trainDataset103.csv", paddingSize, TRAINING_SIZE)
    valSet, valLabels = datasetBuilder3.build2DFeatureSet(sampleSize, "valDataset103.csv", paddingSize, VAL_SIZE)
    trainSet = trainSet.reshape(trainSet.shape[0], paddingSize, paddingSize, 1)
    valSet = valSet.reshape(valSet.shape[0], paddingSize, paddingSize, 1)

    for X in range(1,4): #complete each test 3 times
        model.load_weights('initialWeights.h5')
        trainResults = model.fit(x=trainSet, y=trainLabels, validation_data= (valSet, valLabels),epochs=40, batch_size=64, callbacks = [callback])
        valLoss += trainResults.history['val_loss'][-1]
        valAccuracy += trainResults.history['val_accuracy'][-1]
        trainAccuracy += trainResults.history['accuracy'][-1]
        trainLoss += trainResults.history['loss'][-1]

    results.append([sampleSize, valLoss/3, valAccuracy/3, trainLoss/3, trainAccuracy/3]) #return average of values
    return results


columns = ["Sample Size","Val Loss", "Val Accuracy", "Training Loss", "Training Accuracy"]
results = []#pd.DataFrame(testIndicies, columns=["Test Flow Indices"])

paddingSize = 22

inputShape = (22,22,1)

model = Sequential() #the CNN used for PIM size testing
model.add(Conv2D(32, kernel_size=3, strides=1, activation='relu', input_shape=inputShape, padding='same'))  # C1
model.add(MaxPooling2D(pool_size=(2, 2)))  # S2
model.add(Conv2D(64, kernel_size=5, strides=1, activation='relu', input_shape=inputShape, padding='valid'))  # C3
model.add(MaxPooling2D(pool_size=(2, 2)))  # S4
model.add(Flatten())  # Flatten
model.add(Dense(50, activation='relu'))  # 2 dense layers aka C5
model.add(Dropout(0.2))
model.add(Dense(25,  activation='relu')) #F6
model.add(Dropout(0.2))
model.add(Dense(13, activation='softmax'))
optimizer = tf.keras.optimizers.Adam()  # learning_rate = 0.01 originally
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.save_weights('initialWeights.h5') #save weights so model can be reset after every test"""


for X in range(2, 11): #train model on PIM sizes 2-10 and get results
    results.extend(trainModel(X, model))


resultDf = pd.DataFrame(results, columns=columns)
resultDf.to_csv("Sample_Size_ReducedClassTest.csv", index=False)



