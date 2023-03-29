import pandas as pd

column_names = ["Flow", "Direction","Length", "Arrival Time"]



#Recount and reset flow indicies of a class set so that they start from 1 in the csv file
def flowCounter(flowsList):

    flows = 0
    currentFlow = 0

    for row in flowsList:
        if not (currentFlow == row[0]):
            flows += 1
            currentFlow = row[0]
        row[0] = flows

    return flowsList

#Removes flows with fewer than a given number of packets and writes the remaining flows to a csv file
def removeSmallFlows(folderName, minimumFlowSize):
    print('Doing ', folderName)
    fileName = folderName+ ".csv"
    data = pd.read_csv(fileName, sep=",", header=0) #read in flow data of a class from file
    list = data.values.tolist()

    totalFlows = list[-1][0]
    flows = 0
    currentFlow = 0
    flowsRemoved = 0
    flowLength = 0
    newFlowslist = [[]]
    rowCounter = 0

    for row in list:
        if not (currentFlow == row[0]):
            if(flowLength >= minimumFlowSize): #if flow is bigger than minimum flow size, add it to the new list
                newFlowslist.extend(list[rowCounter - (flowLength):rowCounter])
            else:
                flowsRemoved+=1

            flowLength = 1
            flows += 1
            currentFlow = row[0]
        else:
            flowLength +=1
        rowCounter +=1

    newFlowslist.remove([])
    newFlowslist = flowCounter(newFlowslist) #recount flow indices

    df = pd.DataFrame(newFlowslist, columns = column_names)
    df.to_csv("FlowSize_" + str(minimumFlowSize) + "_"+folderName + ".csv", index=False)

    print(folderName, " Before: ", totalFlows, " flow remaining:", totalFlows-flowsRemoved)
    return flows


classes = {"Chat", "Streaming","VoIP", "P2P", "Browsing", "FT", "VPN Chat", "VPN Streaming","VPN VoIP", "VPN P2P", "VPN FT"}
"""
for X in classes: #Remove flows smaller than 10 packets so that remaining flows have 10 or more packets 
    #fileName = "FlowSize_" + str(10)+ "_"+ X
    removeSmallFlows(X, 10)"""