



"""
Adapted OpenFlow 1.0 L2 learning switch implementation.
Original SimpleSwitch class from: https://github.com/faucetsdn/ryu/blob/master/ryu/app/simple_switch.py
Was customised to allow RYU controller to classify flows with PCNN

Original copyright notice:
 
# Copyright (C) 2011 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.models import Sequential
import time
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_0
from ryu.lib.mac import haddr_to_bin
from ryu.lib.packet import ethernet, packet
from ryu.lib.packet import ipv4
from ryu.lib.packet import ipv6
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.lib.packet import ether_types


class SimpleSwitch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch, self).__init__(*args, **kwargs)
        self.hashmap = dict({}) #holds pims for each flow up until enough pkts are received for classification,
        #ideally entries in the hashmap would be deleted after a certain amount of time to save space
        self.mac_to_port = {} 
        self.classifier = self.getClassifier()

    #Build and compile 2D CNN with the optimised parameters
    def getClassifier(self):

        inputShape = (12,12,1)
        model = Sequential()
        model.add(Conv2D(86, kernel_size=3, strides=1, activation='tanh', input_shape=inputShape, padding='same'))  # C1
        model.add(MaxPooling2D(pool_size=(2, 2)))  # S2
        model.add(Conv2D(86*2, kernel_size=5, strides=1, activation='tanh', input_shape=inputShape, padding='same'))  # C3

        model.add(MaxPooling2D(pool_size=(2, 2)))#S4

        model.add(Flatten())  # Flatten
        for i in range(2):
            model.add(Dense(217, activation='relu'))
            model.add(Dropout(0.05379819490458496))

        model.add(Dense(66, activation='relu'))
        model.add(Dropout(0.05379819490458496))
        model.add(Dense(13, activation='softmax'))

        model.load_weights('Optimisedss6Weights3.h5')

        return model

    #Treats flows as bidirecitonal so adds 2 flow entries for forward and backward directions
    def add_flow(self, datapath, ip_src, ip_dst, ip_proto, srcPort, dstPort, inPort, outPort):
        ofproto = datapath.ofproto
        idleTimeout = 5 
        hardTimeout = 15

        forwardMatch = datapath.ofproto_parser.OFPMatch( 
                    dl_type=ether_types.ETH_TYPE_IP, 
                    nw_proto=ip_proto, nw_src=ip_src, nw_dst=ip_dst, tp_src=srcPort, tp_dst=dstPort)

        backwardMatch = datapath.ofproto_parser.OFPMatch( 
                    dl_type=ether_types.ETH_TYPE_IP, 
                    nw_proto=ip_proto, nw_src=ip_dst, nw_dst=ip_src, tp_src=srcPort, tp_dst=dstPort)

        forwardAction = [datapath.ofproto_parser.OFPActionOutput(outPort)]
        backwardAction = [datapath.ofproto_parser.OFPActionOutput(inPort)]

        #Update switch flow table for forward and backward directions of flow

        modf = datapath.ofproto_parser.OFPFlowMod(
            datapath=datapath, match=forwardMatch, cookie=0,
            command=ofproto.OFPFC_ADD, idle_timeout=idleTimeout, hard_timeout=hardTimeout,
            priority=ofproto.OFP_DEFAULT_PRIORITY,
            flags=ofproto.OFPFF_SEND_FLOW_REM, actions=forwardAction)
        datapath.send_msg(modf)

        modb = datapath.ofproto_parser.OFPFlowMod(
            datapath=datapath, match=backwardMatch, cookie=0,
            command=ofproto.OFPFC_ADD, idle_timeout=idleTimeout, hard_timeout=hardTimeout,
            priority=ofproto.OFP_DEFAULT_PRIORITY,
            flags=ofproto.OFPFF_SEND_FLOW_REM, actions=backwardAction)
        datapath.send_msg(modb)


    #Calculate packet feature vector values and normalise them
    def preprocessPIM(self, pim):
        maxLen = -1
        maxIAT = -1
        lastIAT = 0
        rowCounter = 0
        newPim = np.zeros((12,12)) #make a new pim padded with zeroes

        for X in pim: #calculate features

            if(rowCounter == 0):
                iat = 0
            else:
                iat = X[2] - lastIAT

            lastIAT = X[2]

            if(X[1] > maxLen):
                maxLen = X[1]

            if(iat > maxIAT):
                maxIAT = iat

            newPim[rowCounter+3][4] = X[0]
            newPim[rowCounter+3][5] = X[1]
            newPim[rowCounter+3][6] = iat

            rowCounter+=1
            
        for i in range(6): #normalise values
            newPim[i+3][5] /= maxLen
            newPim[i+3][6] /= maxIAT

        return newPim



    def classifyPIM(self, pim, classifStartTime):
        predTime = time.time_ns()
        prediction = self.classifier.predict(pim.reshape(1, 12, 12, 1)) #classify pim
        endTime = time.time_ns()
        preprocessTime = predTime - classifStartTime
        predictionTime = endTime-predTime
        #prediction.argmax()
        fileData = "\n" + str(preprocessTime) + "," + str(predictionTime)

        recordFile = open('SDNrecord.txt', 'a') #write information to file for analysis
        recordFile.write(fileData)
        recordFile.close()

    #Adds packet to pim hashmap and returns 1 if flow is to be added to flow table else returns 0
    def addPktToHashMap(self, direction, length, arrivalTime, key):
        pktCounter = self.hashmap[key][0]
        pktCounter+=1

        if(pktCounter == 6): #if enough packets acquired do classification
            print("CLASSIFICATION TIME!")
            startTime = time.time_ns()
            pim = self.hashmap[key][1]
            pim.append([direction, length, arrivalTime])
            finalPim = self.preprocessPIM(pim) #process and classify pim
            self.classifyPIM(finalPim, startTime)
            self.hashmap.pop(key) #remove pim entry from hashmap

            return 1 #return 1 as enough packets of this flow have been received

        else: #else if not enough packets received yet, append this packet to pim in hashmap
            self.hashmap[key][0] = pktCounter
            self.hashmap[key][1].append([direction, length, arrivalTime])
            return 0
        

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER) #only acknowledge packet in messages after switch-controller handshake
    def _packet_in_handler(self, ev):
        msg = ev.msg  #data structure containing object of received packet
        datapath = msg.datapath #data structure containing object respresenting switch
        ofproto = datapath.ofproto #agreed openflow protocol

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        ip = pkt.get_protocol(ipv4.ipv4)
        tp = pkt.get_protocol(tcp.tcp)
        proto = ipv4.tcp
        length = len(pkt)
        arrivalTime = time.time_ns()

        if(tp == None):

            if(pkt.get_protocol(udp.udp) != None):
                tp = pkt.get_protocol(udp.udp)
                proto = ipv4.udp
            else: #if no udp or tcp header ignore this packet
                return

        #to figure out the packet direction, 2 keys are tested, with ip source and dest ports swapped
        hashKey1 =  str(proto) + ip.src + ip.dst + str(tp.src_port) + str(tp.dst_port) + "" #make key to store flow packets in the hashmap
        hashKey2 =  str(proto) + ip.dst + ip.src + str(tp.src_port) + str(tp.dst_port) + "" 

        saveToFlowTable = 0

        if(hashKey1 in self.hashmap):
            saveToFlowTable = self.addPktToHashMap(1, length, arrivalTime, hashKey1)
        elif hashKey2 in self.hashmap:
            saveToFlowTable = self.addPktToHashMap(0, length, arrivalTime, hashKey2)
        else:
            self.hashmap[hashKey1]  = [1, [[1, length, arrivalTime]]] #create new pim entry in hashmap

        #Now find which ethernet port to send packet 
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            return
        dst = eth.dst
        src = eth.src

        dpid = datapath.id #get switch id
        self.mac_to_port.setdefault(dpid, {})

        self.logger.info("packet in %s %s %s %s", dpid, src, dst, msg.in_port)

        #learn flow 5 tuple to avoid FLOODing pkt to all ports next time
        self.mac_to_port[dpid][src] = msg.in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst] #get output port for this packet if its known
        else:
            out_port = ofproto.OFPP_FLOOD #otherwise send it out to all ports

        actions = [datapath.ofproto_parser.OFPActionOutput(out_port)]

        # if enough packets received for classification and the output port is known, install a flow in flow table to avoid packet_in next time
        if saveToFlowTable and out_port != ofproto.OFPP_FLOOD:
            self.add_flow(datapath, ip.src, ip.dst, ip.proto, tp.src_port, tp.dst_port, msg.in_port, out_port)

        #Then tell switch to send packet to output port (or all ports in the case of FLOOD action)
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = datapath.ofproto_parser.OFPPacketOut( 
            datapath=datapath, buffer_id=msg.buffer_id, in_port=msg.in_port,
            actions=actions, data=data)
        datapath.send_msg(out)

    #Handle different events
    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def _port_status_handler(self, ev):
        msg = ev.msg
        reason = msg.reason
        port_no = msg.desc.port_no

        ofproto = msg.datapath.ofproto
        if reason == ofproto.OFPPR_ADD:
            self.logger.info("port added %s", port_no)
        elif reason == ofproto.OFPPR_DELETE:
            self.logger.info("port deleted %s", port_no)
        elif reason == ofproto.OFPPR_MODIFY:
            self.logger.info("port modified %s", port_no)
        else:
            self.logger.info("Illeagal port state %s %s", port_no, reason)