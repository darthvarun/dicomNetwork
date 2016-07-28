import numpy as np 
import os
import dicom
import pylab
from matplotlib import pyplot
import sfml as sf
from resizeimage import resizeimage
from PIL import Image
import math
import scipy.misc 

training_images = []
testing_images = []
#dso = dicom.read_file("C:\DicomTraining" + "\\" + "1.dcm")
#dsoarr = dso.pixel_array
#np.reshape(dsoarr, [262144, 1])
#print np.reshape(scipy.misc.imresize(dsoarr, [28, 28]), [784, 1])

def genTrainingData():
	#print "genData has been invoked, bro" #debugging
	for fileName in os.listdir("C:\DicomTraining"):
		ds = dicom.read_file("C:\DicomTraining" + "\\" + fileName)
		dicomArray = ds.pixel_array
		normalized = np.array(dicomArray)
		normalized = np.reshape(normalized, [262144, 1])
		#normalized = normalized/1024.0
		training_images.append([resizeImage(normalized, 28, 28), fileName[0: int(len(fileName)-4)]])
	#print "We will see this if genData succesfully loads the images, bro" #debugging 2.0
	return training_images

def genTestingData():
	#print "genData has been invoked, bro" #debugging
	for fileName2 in os.listdir("C:\DicomTesting"):
		ds2 = dicom.read_file("C:\DicomTesting" + "\\" + fileName2)
		dicomArray2 = ds2.pixel_array
		normalized2 = np.array(dicomArray2)
		normalized2 = np.reshape(normalized2, [262144, 1])
		#normalized = normalized/1024.0
		testing_images.append([resizeImage(normalized2, 28, 28), fileName2[0: int(len(fileName2)-4)]])
	#print "We will see this if genData succesfully loads the images, bro" #debugging 2.0
	return testing_images


def resizeImage(pixelArray, newWidth, newHeight):
	print "resizeImage is being invoked fam!!!"	#even more debugging
	return np.reshape(scipy.misc.imresize(pixelArray, [newWidth, newHeight]), [newWidth*newHeight, 1])

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
	
	"""a = np.empty(784)
	np.reshape(pixelArray, [w1*h1, 1])
	resized =np.zeros(w2*w2, dtype=object)
	x_ratio = int(w1/float(w2))
	y_ratio = int(h1/float(h2))
	px = 0
	py = 0
	for i in xrange(h2):
		for j in xrange(w2):
			px = math.floor(j*x_ratio)
			py = math.floor(i*y_ratio)
			resized[(i*w2)+j] = pixelArray[int((py*w1)+px)]
	for value in resized:
		for number in value:			
			np.append(a, value)
	return np.reshape(a, [784, 1])
	img = Image.fromarray(a)
	img.show()"""

"""def resizeTestingImage(pixelArray, w1, h1, w2, h2):
	#print "resizeImage is being invoked fam!!!"	#even more debugging
	b = np.empty(784)
	np.reshape(pixelArray, [w1*h1, 1])
	resized2 =np.zeros(w2*w2, dtype=object)
	x_ratio2 = int(w1/float(w2))
	y_ratio2 = int(h1/float(h2))
	px2 = 0
	py2 = 0
	for i in xrange(h2):
		for j in xrange(w2):
			px2 = math.floor(j*x_ratio2)
			py2 = math.floor(i*y_ratio2)
			resized2[(i*w2)+j] = pixelArray[int((py2*w1)+px2)]
	for value in resized2:
		for number in value:			
			np.append(b, value)
	return np.reshape(b, [784, 1])"""








	