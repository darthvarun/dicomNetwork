This project uses Neural Networks and Machine Learning techniques to analyze DICOM medical images, with an attempted functionality of a craniocaudal z-axis classifier. 

***PROJECT SETUP***

In order to set up the project, ensure that you have two separate folders somewhere on your hard drive with the DICOM images you intend to train and test with. Then, go to the GenTrainingData program and adjust the paths within the genTraningData() method and genTestingData() method respectively to match the paths of your files. 


Within the runNetwork.py program, for the line: 

net = network.Network([784, 30, 10])

Leave the first and third parameters unchanged as that will probably cause a numpy mismatch with the ndarray dimensions. You can adjust the second parameter, which controls the number of hidden layers within the network, however.

Finally, within the line: 

net.SGD(training_data, 30, 5, 3.0, test_data = testing_data)

All three parameters are adjustable. The first parameter controls the number of epochs that you want to run the network for, followed by the mini batch size (the size of each individual batch that you want to apply the gradient descent algorithm to) and finally the desired learning rate of the network. 

