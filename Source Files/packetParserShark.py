import os
import sys

import pandas as pd
import pyshark

#Feature extractor using pyshark

def readPcap(filePath):
    pcap = pyshark.FileCapture(filePath)
    counter = 0
    numPackets = len(pcap)
    flowsList = []

    flows = 0

    lastSourceIp = ""  # last source ip, used to determine if packet part of previous flow
    lastDestIp = ""  # last destination ip
    lastSourcePort = 0
    lastDestPort = 0
    lastWasTCP = False #indicates if last packet was a TCP packet

    for pkt in pcap:

        samePortProtocol = False
        sameIps = False

        if('IP' not in pkt): #ignore any packets without an accessible IP header
            continue

        if('TCP' not in pkt and 'UDP' not in pkt): #if no udp or tcp header ignore this packet
            continue

        if ('TCP' in pkt.transport_layer):  #determine if tranport layer protocol being used is UDP or TCP
            if(lastWasTCP):
                samePortProtocol = True
            lastWasTCP = True
        else:
            if(not lastWasTCP):
                samePortProtocol = True
            lastWasTCP = False

        currentSPort = pkt[pkt.transport_layer].srcport
        currentDPort = pkt[pkt.transport_layer].dstport

        if(pkt.ip.src == lastSourceIp and pkt.ip.dst == lastDestIp and currentSPort == lastSourcePort and currentDPort == lastDestPort):
            direction = 1
            sameIps = True
        elif(pkt.ip.src == lastDestIp and pkt.ip.dst == lastSourceIp and currentSPort == lastDestPort and currentDPort == lastSourcePort):
            direction = -1
            sameIps = True
        else:
            direction = 1

        if not (sameIps and samePortProtocol): #if doesn't belong to same flow, initialise flow values
            flows += 1
            lastSourceIp = pkt.ip.src
            lastDestIp = pkt.ip.dst
            lastSourcePort = currentSPort
            lastDestPort = currentDPort

        flowsList.append([flows, direction,  pkt.length,  pkt.sniff_timestamp])  # add packet info to dataframe
        counter+=1

        if counter % 1000 == 0:
            print(counter,"/",numPackets)

    return flowsList

column_names = ["Flow", "Direction", "Length", "Arrival Time"]

if __name__ == '__main__':
    folderNames = sys.argv[1:]
    directory = "D:\\Documents\\VS Code Projects\\TYP Dataset\\Pcaps" #all pcaps are in this directory

    for folder in folderNames:
        print('Starting ', folder)
        folderDirectory = directory + '\\' + folder
        fileNames = os.listdir(folderDirectory)  # get pcap files in folders
        flowsList = []
        filepath = folderDirectory.replace('\\', '/')

        for x in fileNames: #read pcaps in each file
            result = readPcap(filepath + '/' + x)
            flowsList.extend(result)

        df = pd.DataFrame(flowsList, columns=column_names)  # make dataframe and write flow data to csv file
        dfFileName = folder + '.csv'
        df.to_csv(dfFileName, index=False)
        print("Folder ", folder, " is complete")
