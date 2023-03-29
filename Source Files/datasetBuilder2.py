import math

import numpy as np
import pandas as pd
import random

NUMBER_OF_SAMPLES = 15065
TRAINING_SIZE = math.ceil(NUMBER_OF_SAMPLES *0.7)
VAL_SIZE = math.floor(NUMBER_OF_SAMPLES *0.1)
TEST_SIZE = math.floor(NUMBER_OF_SAMPLES *0.2)

column_names = ["Flow", "Pkt Dir","Pkt len", "IAT","Label"]
trainSet = []
valSet = []
testSet = []
testIndicies = []

#Get a select number of random flows from each traffic class and writes them to a trainset, valset and testset as 70:10:20 split
def getRandomSamples(numSamples, list, label):
    maxFlow = list[-1][0]

    if(maxFlow > numSamples): #get random sample if more flows than sample number
        sampleFlows = random.sample(range(1, int(maxFlow+1)), numSamples) #gets flow numbers than samples will be taken from
        sampleFlows.sort()
    else:
        sampleFlows = range(1,int(maxFlow+1))  #otherwise just use all flows in this class

    trainList= [] #list for training set
    testList  = [] #list for testing set
    valList = [] #list for validation set
    sampleCounter = 0
    rowCounter = 0 #used to index of flow in test set so that they can be traced back to the pcap they originated from
    currentSampleNumber = sampleFlows[sampleCounter]

    for row in list:

        if(int(row[0]) in sampleFlows):
            row.append(label) #add label column

            if not (currentSampleNumber == row[0]):
                sampleCounter+=1
                currentSampleNumber = row[0]
            if (sampleCounter >= TRAINING_SIZE + TEST_SIZE):  # add to training list, test list or val list based on how many samples added
                valList.append(row)
            elif sampleCounter >= TRAINING_SIZE:
                testList.append(row[:])
                testIndicies.append(rowCounter)
            else:
                trainList.append(row)
        elif row[0] > sampleFlows[-1]:
            break

        rowCounter+=1

    print(sampleCounter)
    return trainList, testList, valList

#Builds a test set and training set using random sampling of the dataset
def getSamples(folderName, sampleSize):
    folder = "FlowSize 10 Sampling\\"
    fileName = folder +"FlowSize_" + str(sampleSize) + "_" + folderName+ ".csv"
    data = pd.read_csv(fileName, sep=",", header=0)
    list = data.values.tolist()

    return getRandomSamples(NUMBER_OF_SAMPLES,list, folderName)


#select the flows of a given class for each dataset
classes = {"Browsing","Streaming","VoIP","FT",  "VPN Streaming","VPN VoIP", "VPN FT", "P2P", "VPN P2P"}
for X in classes:
    print("Doing ", X)
    trainList, testList, valList = getSamples(X, 10)
    trainSet.extend(trainList)
    testSet.extend(testList)
    valSet.extend(valList)
    print(X)

#Write training, testing and validation sets to file
trainingSet = pd.DataFrame(trainSet, columns=column_names)
testingSet = pd.DataFrame(testSet, columns=column_names)
valSet = pd.DataFrame(valSet, columns=column_names)
testindexes = pd.DataFrame(testIndicies, columns=["Test Flow Indices"])
trainingSet.to_csv("trainDataset103.csv", index=False)
valSet.to_csv("valDataset103.csv", index=False)
testingSet.to_csv("testDataset103.csv", index=False)
testindexes.to_csv("testIndices103.csv", index=False)