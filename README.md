# DL-on-Network-Classification

Officially titled: Applying Deep Learning to the classification of VPN and non-VPN encrypted network traffic

This was a project that I completed for my dissertation in my final year of university.

Due to the rapid increase in internet traffic, network traffic classification is required for Quality
of Service (QoS) management. The use of Virtual Private Networks (VPNs) and encryption
has made this much more difficult, as existing methods such as deep packet inspection have
become futile. Hence, many researchers have begun applying Deep Learning to this problem.
This project employs a payload-independent method in which traffic flows are represented as
Pseudo Image Matrices (PIMs), and input into a Convolutional Neural Network to categorize
them into different classes i.e., Browsing, Streaming and Email. Unlike many classifiers in
current literature, only 6 packets of a flow are needed to identify its class, therefore facilitating
real-time classification. To address the difficulties of classifying encrypted traffic, this project
focuses on the classification of VPN and non-VPN encrypted traffic. Using the UNB ISCX
VPN-nonVPN dataset, the developed classifier achieves an accuracy of 70.9%, and takes
approximately 39ms to classify a flow after the arrival of the 6th packet.
