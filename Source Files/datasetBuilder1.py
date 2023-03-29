import pandas as pd
from removeSmallFlows import flowCounter

column_names = ["Flow", "Direction","Length", "Arrival Time"]

#Requires flows to have already have been numbered 1,2,3,4....
#Performs fixed step sampling
def fixedStepSize(sampleSize, className):
    fileName = "FlowSize_" + str(sampleSize)+ "_"+ className + ".csv"
    data = pd.read_csv(fileName, sep=",", header=0)
    list = data.values.tolist() #convert to list

    ssCounter = 0 #sample size counter
    currentSampleIndex = 1
    flows = 0
    currentFlow = 0

    for row in list: #iterate through packets in each flow
        if not (currentFlow == row[0]): #if packet does not belong to same flow as last flow
            currentSampleIndex = 1
            currentFlow = row[0]
            flows += 1 #increment flow count
            ssCounter = 1 #reset sample count

        else: #else if part of same flow
            ssCounter += 1  # increment sampleSize counter

            if currentSampleIndex > 1: #if this flow contains more than 1 sample
                row[0] = currentFlow + currentSampleIndex  # set flow index to new value that packets in new sample have

            if(ssCounter > sampleSize): #if enough for multiple samples in this flow, divide flow into another flow
                ssCounter = 1
                flows+=1
                currentSampleIndex += 1
                row[0] = currentFlow + currentSampleIndex #set flow index to new value that packets in new sample will have

    list = flowCounter(list) #recount flows in list
    newFlowCount = list[-1][0]

    df = pd.DataFrame(list, columns = column_names)
    df.to_csv(fileName, index=False) #write new flows to csv file

    return newFlowCount #return new flow count



#Accurately sets flow count so that it is ordered flow count starts from 1 to ...
def countFlows(folderName):
    print('Doing ', folderName)
    fileName = folderName+ ".csv"
    data = pd.read_csv(fileName, sep=",", header=0)
    list = data.values.tolist()

    flows = 0
    currentFlow = 0

    for row in list:
        if not (currentFlow == row[0]):
            flows += 1
            currentFlow = row[0]
        row[0] = flows

    return flows


classes = { "Streaming","VoIP", "P2P", "Browsing", "FT", "VPN Streaming","VPN VoIP", "VPN P2P", "VPN FT"}
for X in classes:
    fixedStepSize(6, 10)
