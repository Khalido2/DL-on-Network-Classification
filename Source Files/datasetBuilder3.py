import math
import random

import numpy
import numpy as np

import pandas as pd

#Ordinal encoding for class names
classCodes = dict({"Browsing": 0, "Streaming": 1, "VoIP":2, "P2P": 3,"FT": 4, "VPN VoIP": 5, "VPN Streaming": 6, "VPN FT": 7, "VPN P2P": 8})
numClasses = len(classCodes)

#From a dataset containing full flows returns the 3D dataset of PIMs of the first K (sample size) packets of each flow
#padlength = final length of PIM with padding
def build2DFeatureSet(sampleSize, fileName ,padLength, numData):
    data = pd.read_csv(fileName, sep=",", header=0)
    arr = data.to_numpy()

    featureSet = numpy.zeros(shape=(numData*numClasses, padLength, padLength))
    labelSet = numpy.zeros(numData*numClasses)
    sampleCounter = 0
    counter = 0
    rowCounter = 0
    lastFlow = 0
    lastLabel = 0
    flowStart = 0

    for row in arr:

        if row[0] == lastFlow :
            if counter < sampleSize:
                counter += 1
        else:
            #Write features of flow that just passed
            if(counter > 0):
                featureSet[sampleCounter] = calcFeatures(arr[flowStart:flowStart+counter], padLength)  # when correct number of samples taken, calculate features for this flow
                labelSet[sampleCounter] = classCodes[lastLabel]
                sampleCounter += 1

            lastFlow = row[0]
            lastLabel = row[4]
            flowStart = rowCounter
            counter = 1

        rowCounter+=1

    #add last flow in dataset
    featureSet[sampleCounter] = calcFeatures(arr[flowStart:flowStart + counter], padLength)  # when correct number of samples taken, calculate features for this flow
    labelSet[sampleCounter] = classCodes[lastLabel]

    labelSet = labelSet.astype(int) #ensure all labels are integer values

    return featureSet, labelSet


#return set of features for given flow
def calcFeatures(flow, padLength):

    #Calculate necessary padding
    hPad = padLength - 3
    vPad = padLength - len(flow)
    hPadLeft = [float(0)] * math.floor(hPad/2)
    hPadRight = [float(0)] * math.ceil(hPad/2)
    vPadTop = [[float(0)]*(hPad+3)]* math.floor(vPad/2)
    vPadBottom = [[float(0)]*(hPad+3)] * math.ceil(vPad/2)

    featureList = []
    maxIAT = -1
    maxPktLen = -1
    IAT = 0
    lastIAT = 0
    rowCounter = 0

    for row in flow:

        if(rowCounter == 0): #calc IAT
            IAT = 0
        else:
            IAT = row[3] - lastIAT

        lastIAT = row[3]

        if(row[1] < 0): #normalise negative direction to 0, positive direction = 1
            row[1] = 0

        innerSample = hPadLeft.copy()
        innerSample.append(row[1])
        innerSample.append(row[2])
        innerSample.append(IAT)
        innerSample.extend(hPadRight.copy())
        featureList.append(innerSample)
        #featureList.append([row[1], row[2], IAT])

        if IAT > maxIAT: #calc max IAT
            maxIAT = IAT

        if row[2] > maxPktLen:
            maxPktLen = row[2]

        rowCounter+=1

    # add vertical padding
    for row in vPadTop:
        featureList.insert(0,row)

    featureList.extend(vPadBottom)

    iatIndex = 2 + len(hPadLeft)
    lenIndex = 1+len(hPadLeft)

    #Normalise Values
    for row in featureList:
        if(maxIAT != 0):
            row[iatIndex] = row[iatIndex]/maxIAT

        row[lenIndex] = row[lenIndex]/maxPktLen

    return featureList

