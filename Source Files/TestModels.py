import math
from ax.utils.notebook.plotting import render
import pandas as pd
import tensorflow as tf
from ax.service.ax_client import AxClient
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.models import Sequential
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import datasetBuilder3
from kerasOptimizerSettings import get_keras_model
from kerasAxExperiment import evaluate

NUMBER_OF_SAMPLES = 15065 # number of PIMs in each class
TRAINING_SIZE = math.ceil(NUMBER_OF_SAMPLES *0.7)
TEST_SIZE = math.floor(NUMBER_OF_SAMPLES *0.2)
VAL_SIZE = math.floor(NUMBER_OF_SAMPLES *0.1)

sampleSize = 6
paddingSize = 12
labels = ["Browsing", "Streaming", "VoIP", "P2P","FT", "VPN VoIP", "VPN Streaming", "VPN FT", "VPN P2P"]

#Tests the model on test set, and calculates a confusion matrix and other performance metrics, this is written to a csv file
def testModel(model, testSet, testLabels):
    testSet = testSet.reshape(testSet.shape[0], paddingSize, paddingSize, 1)
    predictions = model.predict(testSet)
    results = []
    accuracy =  accuracy_score(testLabels, predictions.argmax(axis=1))
    precisions = precision_score(testLabels, predictions.argmax(axis=1), average=None)
    precision = precision_score(testLabels, predictions.argmax(axis=1), average='weighted')
    recalls = recall_score(testLabels, predictions.argmax(axis=1), average=None)
    recall = recall_score(testLabels, predictions.argmax(axis=1), average='weighted')
    f1scores = f1_score(testLabels, predictions.argmax(axis=1), average=None)
    f1score = f1_score(testLabels, predictions.argmax(axis=1), average='weighted')

    """print(accuracy)
    print('Precision: ', precision_score(testLabels, predictions.argmax(axis=1), average=None))
    print('Accuracy: ', accuracy_score(testLabels, predictions.argmax(axis=1)))
    print('Recall: ', recall_score(testLabels, predictions.argmax(axis=1), average=None))
    print('F1 score: ', f1_score(testLabels, predictions.argmax(axis=1), average=None))"""

    disp = ConfusionMatrixDisplay.from_predictions(testLabels, predictions.argmax(axis=1), display_labels=labels, cmap="Blues", xticks_rotation=45)
    disp.plot()
    plt.show()

    for X in range (0, 9): #fill list to write to the csv file
        results.append([precisions[X], recalls[X], f1scores[X]])

    results.append([precision, recall, f1score])
    df = pd.DataFrame(results, columns=["Precision", "Recall", "F1 score"])
    df.to_csv("PerformanceResults.csv", index=False)


#If weights havent been saved after training, build model, train it and save weights
def buildModel(parameterization, trainSet, valSet, trainLabels, valLabels):
    model = get_keras_model(parameterization.get('numFilters'),
                            parameterization.get('firstFilterSize'),
                            parameterization.get('secondFilterSize'),
                            parameterization.get('numNeuronsDense'),
                            parameterization.get('numDense'),
                            parameterization.get('numNeuronsDenseLast'),
                            parameterization.get('dropout'),
                            parameterization.get('convolution_activation'),
                            parameterization.get('dense_activation')) #get keras model with these parameters

    numEpochs = parameterization.get('epochs')
    learningRate = parameterization.get('learningRate')
    batchSize = parameterization.get('batchSize')

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                patience=3)  # stop training early if loss doesn't improve in 3 epochs

    optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)


    trainSet = trainSet.reshape(trainSet.shape[0], paddingSize, paddingSize, 1)
    valSet = valSet.reshape(valLabels.shape[0], paddingSize, paddingSize, 1)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=trainSet, y=trainLabels, validation_data=(valSet, valLabels), batch_size=batchSize,
                    epochs=numEpochs)
    model.save_weights('Optimisedss6Weights3.h5')


#Make models
trainSet, trainLabels = datasetBuilder3.build2DFeatureSet(sampleSize, "trainDataset103.csv", paddingSize, TRAINING_SIZE)
testSet, testLabels = datasetBuilder3.build2DFeatureSet(sampleSize, "testDataset103.csv", paddingSize, TEST_SIZE)
valSet, valLabels = datasetBuilder3.build2DFeatureSet(sampleSize, "valDataset103.csv", paddingSize, VAL_SIZE)

newExperiment = 'optimisedCNNss6Accuracy2.json'
axClient = AxClient.load_from_json_file(filepath=newExperiment) #load optimisation experiment
render(axClient.get_optimization_trace())

best_parameters, values = axClient.get_best_parameters()
#balancedModel = buildModel(best_parameters, trainSet, valSet, trainLabels, valLabels)
balancedModel = get_keras_model(best_parameters.get('numFilters'),
                            best_parameters.get('firstFilterSize'),
                            best_parameters.get('secondFilterSize'),
                            best_parameters.get('numNeuronsDense'),
                            best_parameters.get('numDense'),
                            best_parameters.get('numNeuronsDenseLast'),
                            best_parameters.get('dropout'),
                            best_parameters.get('convolution_activation'),
                            best_parameters.get('dense_activation'))  # get keras model with optimised architecture

balancedModel.load_weights('Optimisedss6Weights3.h5')#""" #load the trained weights of the model
testModel(balancedModel, testSet, testLabels)

