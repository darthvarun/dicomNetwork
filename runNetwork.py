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


print "If you see asterisks below, the program is loading the DICOM data"
training_data = GenTrainingData.genTrainingData()
print "Training Data has been loaded!"
testing_data = GenTrainingData.genTestingData()
print "Testing Data has been loaded!"
#print training_data[0]
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 5, 3.0, test_data = testing_data)
