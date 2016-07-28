import numpy as np 
import os
import dicom
import pylab
from matplotlib import pyplot
import sfml as sf
from resizeimage import resizeimage
from PIL import Image
import math 
import GenTrainingData
import network

print "Hello is this even working"
training_data = GenTrainingData.genTrainingData()
testing_data = GenTrainingData.genTestingData()
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data = testing_data)
#print training_data[0]
#print testing_data[0]
#print "this is the division between training and testing lmao"
#print testing_data[0]
