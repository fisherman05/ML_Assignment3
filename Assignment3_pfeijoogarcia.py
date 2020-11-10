# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 18:07:23 2020 (I read the guideline and picked the dataset way before ^-^)
Dataset: Seeds. Available at: http://archive.ics.uci.edu/ml/datasets/seeds#
@author: Pedro Guillermo Feijoo Garcia
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#original lists
seeds = []
labels = []

#training, validation, and testing setups

""" Data Input Reading """
def readInputData():
    global seeds
    global labels
    path = "./data/seeds.data"
    file = open(path, 'r') 
    lines = file.readlines() 
    
    for line in lines:
        data = line.split(";")
        area = float(data[0])
        kernel = float(data[1])
        
        #Acceptability is the class label
        label = int(data[2])
        labels.append(label)
        #I do it like this, so that I can use the KNNClassifier correctly according to python's definition
        #of X
        seed = [area, kernel]
        seeds.append(seed)
    
    seeds = np.array(seeds)
    labels = np.array(labels)
        

""" Datasets' Arrangement Functions """
#I  create my Testing and Training+Validation datasets
#This code relates to the code I wrote for the second assignment
def splitTestingFromTrainingData():
    folds = KFold(n_splits=5, shuffle=True)
    trainSeeds = []
    testSeeds = []
    trainLabels = []
    testLabels = []
    
    for train_i, validate_i in folds.split(labels):
        seed_train, seed_validate = seeds[train_i], seeds[validate_i]
        label_train, label_validate = labels[train_i], labels[validate_i]
        
        trainSeeds.append(seed_train)
        trainLabels.append(label_train)
        testSeeds.append(seed_validate)
        testLabels.append(label_validate)
    
    return (trainSeeds[0],testSeeds[0],trainLabels[0], testLabels[0])

#I arrange 5 folds from the training+validation data
#This code relates to the code I wrote for the second assignment
def generateFolds(trainValidation_seeds, trainValidation_labels):
    folds = KFold(n_splits=5, shuffle=True)
    trainSeeds = []
    validateSeeds = []
    trainLabels = []
    validateLabels = []
    for train_i, validate_i in folds.split(trainValidation_labels):
        seed_train, seed_validate = trainValidation_seeds[train_i], trainValidation_seeds[validate_i]
        label_train, label_validate = trainValidation_labels[train_i], trainValidation_labels[validate_i]
        
        trainSeeds.append(seed_train)
        trainLabels.append(label_train)
        validateSeeds.append(seed_validate)
        validateLabels.append(label_validate)
    
    return (trainSeeds,validateSeeds,trainLabels, validateLabels)

""" K-NN Classifiers """
euclideanClassifier_Btree_uniform_KA = KNeighborsClassifier(n_neighbors=3, algorithm = 'ball_tree', weights='uniform', p=2)
euclideanClassifier_Btree_uniform_KB = KNeighborsClassifier(n_neighbors=20, algorithm = 'ball_tree', weights='uniform', p=2)
euclideanClassifier_Btree_uniform_KC = KNeighborsClassifier(n_neighbors=70, algorithm = 'ball_tree', weights='uniform', p=2)

manhattanClassifier_Btree_distance_KA = KNeighborsClassifier(n_neighbors=3, algorithm = 'ball_tree', weights='distance', p=1)
manhattanClassifier_Btree_distance_KB = KNeighborsClassifier(n_neighbors=20, algorithm = 'ball_tree', weights='distance', p=1)
manhattanClassifier_Btree_distance_KC = KNeighborsClassifier(n_neighbors=70, algorithm = 'ball_tree', weights='distance', p=1)

euclideanClassifier_KDtree_uniform_KA = KNeighborsClassifier(n_neighbors=3, algorithm = 'kd_tree', weights='uniform', p=2)
euclideanClassifier_KDtree_uniform_KB = KNeighborsClassifier(n_neighbors=20, algorithm = 'kd_tree', weights='uniform', p=2)
euclideanClassifier_KDtree_uniform_KC = KNeighborsClassifier(n_neighbors=70, algorithm = 'kd_tree', weights='uniform', p=2)

manhattanClassifier_KDtree_distance_KA = KNeighborsClassifier(n_neighbors=3, algorithm = 'kd_tree', weights='distance', p=1)
manhattanClassifier_KDtree_distance_KB = KNeighborsClassifier(n_neighbors=20, algorithm = 'kd_tree', weights='distance', p=1)
manhattanClassifier_KDtree_distance_KC = KNeighborsClassifier(n_neighbors=70, algorithm = 'kd_tree', weights='distance', p=1)


def labelsPrediction(model, setup):
    X_training = setup[0]
    y_training = setup[2]
    X_test = setup[1]
    y_true = setup[3]
    y_predicted = None
    
    model.fit(X_training, y_training)
    y_predicted= model.predict(X_test)
    
    return (X_training, y_training, X_test, y_predicted, y_true)

def accuracyCalculator(y_true, y_predicted):
    return accuracy_score(y_true, y_predicted)



""" Normalization Algorithms """
def minMaxNormalization(inputSeeds):
    normalizedSeeds = []
    minArea = None
    maxArea = None
    
    minKernel = None
    maxKernel = None
    
    #figure out the min and max of the feature: area
    for seed in inputSeeds:
        if(minArea == None):
            minArea = seed[0]
        elif(minArea > seed[0]):
            minArea = seed[0]
    
    for seed in inputSeeds:
        if(maxArea == None):
            maxArea = seed[0]
        elif(maxArea < seed[0]):
            maxArea = seed[0]
            
    #figure out the min and max of the feature: kernell
    for seed in inputSeeds:
        if(minKernel == None):
            minKernel = seed[1]
        elif(minKernel > seed[1]):
            minKernel = seed[1]
    
    for seed in inputSeeds:
        if(maxKernel == None):
            maxKernel = seed[1]
        elif(maxKernel < seed[1]):
            maxKernel = seed[1]
    
    #normalize each feature per seed
    for seed in inputSeeds:
        area = (seed[0] - minArea)/(maxArea-minArea)
        kernel = (seed[1] - minKernel)/(maxKernel-minKernel)
        normalized = [area, kernel]
        normalizedSeeds.append(normalized)
    
    return np.array(normalizedSeeds)
    
    
def zScoreNormalization(inputSeeds):
    normalizedSeeds = []
    meanArea = None
    stDevArea = None
    
    meanKernel = None
    stDevKernel = None
    
    areas = []
    kernels = []

    for seed in inputSeeds:
        areas.append(seed[0])
        kernels.append(seed[1])
    
    meanArea = np.mean(areas)
    meanKernel = np.mean(kernels)
    stDevArea = np.std(areas)
    stDevKernel = np.std(kernels)
    
    
    
    #normalize each feature per seed
    for seed in inputSeeds:
        area = (seed[0] - meanArea)/stDevArea
        kernel = (seed[1] - meanKernel)/stDevKernel
        normalized = [area, kernel]
        normalizedSeeds.append(normalized)
    
    return np.array(normalizedSeeds)    


""" Code Execution """
#----------------Input Reading --------------------------------
"""
readInputData()
"""
#----------------Data Preprocessing --------------------------------
#the code here relates to the code I wrote for the second assignment
"""
trainValidation_seeds, test_seeds, trainValidation_labels, test_labels = splitTestingFromTrainingData()
np.save("./data/trainValidation_seeds", trainValidation_seeds)
np.save("./data/test_seeds", test_seeds)
np.save("./data/trainValidation_labels", trainValidation_labels)
np.save("./data/test_labels", test_labels)
"""


trainValidation_seeds = np.load("./data/trainValidation_seeds.npy")
test_seeds = np.load("./data/test_seeds.npy")
trainValidation_labels = np.load("./data/trainValidation_labels.npy")
test_labels = np.load("./data/test_labels.npy")

#----------------Data Normalization --------------------------------
#Min-Max
minMaxSeeds_trainValidation = minMaxNormalization(trainValidation_seeds.tolist())
minMaxSeeds_test = minMaxNormalization(test_seeds.tolist())
#I organize the different configurations for training and validation
#the code here relates to the code I wrote for the second assignments

"""
(trainSeeds, validateSeeds, trainLabels, validateLabels) = generateFolds(minMaxSeeds_trainValidation, trainValidation_labels)
np.save("./data/trainSeeds", trainSeeds)
np.save("./data/validateSeeds", validateSeeds)
np.save("./data/trainLabels", trainLabels)
np.save("./data/validateLabels", validateLabels)
"""

trainSeeds = np.load("./data/trainSeeds.npy")
validateSeeds = np.load("./data/validateSeeds.npy")
trainLabels = np.load("./data/trainLabels.npy")
validateLabels = np.load("./data/validateLabels.npy")


minMax_setup1 = (trainSeeds[0], validateSeeds[0], trainLabels[0], validateLabels[0])
minMax_setup2 = (trainSeeds[1], validateSeeds[1], trainLabels[1], validateLabels[1])
minMax_setup3 = (trainSeeds[2], validateSeeds[2], trainLabels[2], validateLabels[2])
minMax_setup4 = (trainSeeds[3], validateSeeds[3], trainLabels[3], validateLabels[3])
minMax_setup5 = (trainSeeds[4], validateSeeds[4], trainLabels[4], validateLabels[4])


#Training and Validation: Cross Validation Multiple Configurations
#This evaluation is to determine the parameter K: we keep constant everything else
euclidean_setup1_configAAA = labelsPrediction(euclideanClassifier_Btree_uniform_KA, minMax_setup1)
euclidean_setup2_configAAA = labelsPrediction(euclideanClassifier_Btree_uniform_KA, minMax_setup2)
euclidean_setup3_configAAA = labelsPrediction(euclideanClassifier_Btree_uniform_KA, minMax_setup3)
euclidean_setup4_configAAA = labelsPrediction(euclideanClassifier_Btree_uniform_KA, minMax_setup4)
euclidean_setup5_configAAA = labelsPrediction(euclideanClassifier_Btree_uniform_KA, minMax_setup5)
euclidean_setup1_configAAB = labelsPrediction(euclideanClassifier_Btree_uniform_KB, minMax_setup1)
euclidean_setup2_configAAB = labelsPrediction(euclideanClassifier_Btree_uniform_KB, minMax_setup2)
euclidean_setup3_configAAB = labelsPrediction(euclideanClassifier_Btree_uniform_KB, minMax_setup3)
euclidean_setup4_configAAB = labelsPrediction(euclideanClassifier_Btree_uniform_KB, minMax_setup4)
euclidean_setup5_configAAB = labelsPrediction(euclideanClassifier_Btree_uniform_KB, minMax_setup5)
euclidean_setup1_configAAC = labelsPrediction(euclideanClassifier_Btree_uniform_KC, minMax_setup1)
euclidean_setup2_configAAC = labelsPrediction(euclideanClassifier_Btree_uniform_KC, minMax_setup2)
euclidean_setup3_configAAC = labelsPrediction(euclideanClassifier_Btree_uniform_KC, minMax_setup3)
euclidean_setup4_configAAC = labelsPrediction(euclideanClassifier_Btree_uniform_KC, minMax_setup4)
euclidean_setup5_configAAC = labelsPrediction(euclideanClassifier_Btree_uniform_KC, minMax_setup5)

AC_setup1_configAAA = accuracyCalculator(euclidean_setup1_configAAA[4], euclidean_setup1_configAAA[3]) 
AC_setup2_configAAA = accuracyCalculator(euclidean_setup2_configAAA[4], euclidean_setup2_configAAA[3]) 
AC_setup3_configAAA = accuracyCalculator(euclidean_setup3_configAAA[4], euclidean_setup3_configAAA[3]) 
AC_setup4_configAAA = accuracyCalculator(euclidean_setup4_configAAA[4], euclidean_setup4_configAAA[3]) 
AC_setup5_configAAA = accuracyCalculator(euclidean_setup5_configAAA[4], euclidean_setup5_configAAA[3]) 

AC_configAAA = np.average([AC_setup1_configAAA, AC_setup2_configAAA, AC_setup3_configAAA, AC_setup4_configAAA, AC_setup5_configAAA])

AC_setup1_configAAB = accuracyCalculator(euclidean_setup1_configAAB[4], euclidean_setup1_configAAB[3]) 
AC_setup2_configAAB = accuracyCalculator(euclidean_setup2_configAAB[4], euclidean_setup2_configAAB[3]) 
AC_setup3_configAAB = accuracyCalculator(euclidean_setup3_configAAB[4], euclidean_setup3_configAAB[3]) 
AC_setup4_configAAB = accuracyCalculator(euclidean_setup4_configAAB[4], euclidean_setup4_configAAB[3]) 
AC_setup5_configAAB = accuracyCalculator(euclidean_setup5_configAAB[4], euclidean_setup5_configAAB[3]) 

AC_configAAB = np.average([AC_setup1_configAAB, AC_setup2_configAAB, AC_setup3_configAAB, AC_setup4_configAAB, AC_setup5_configAAB])


AC_setup1_configAAC = accuracyCalculator(euclidean_setup1_configAAC[4], euclidean_setup1_configAAC[3]) 
AC_setup2_configAAC = accuracyCalculator(euclidean_setup2_configAAC[4], euclidean_setup2_configAAC[3]) 
AC_setup3_configAAC = accuracyCalculator(euclidean_setup3_configAAC[4], euclidean_setup3_configAAC[3]) 
AC_setup4_configAAC = accuracyCalculator(euclidean_setup4_configAAC[4], euclidean_setup4_configAAC[3]) 
AC_setup5_configAAC = accuracyCalculator(euclidean_setup5_configAAC[4], euclidean_setup5_configAAC[3]) 

AC_configAAC = np.average([AC_setup1_configAAC, AC_setup2_configAAC, AC_setup3_configAAC, AC_setup4_configAAC, AC_setup5_configAAC])


print('Config AAA: Setup 1-' + str(AC_setup1_configAAA) + ' Setup 2-'+ str(AC_setup2_configAAA) + ' Setup 3-'+ str(AC_setup3_configAAA) + ' Setup 4-'+ str(AC_setup4_configAAA) + ' Setup 5-'+ str(AC_setup5_configAAA))
print('Config AAB: Setup 1-' + str(AC_setup1_configAAB) + ' Setup 2-'+ str(AC_setup2_configAAB) + ' Setup 3-'+ str(AC_setup3_configAAB) + ' Setup 4-'+ str(AC_setup4_configAAB) + ' Setup 5-'+ str(AC_setup5_configAAB))
print('Config AAC: Setup 1-' + str(AC_setup1_configAAC) + ' Setup 2-'+ str(AC_setup2_configAAC) + ' Setup 3-'+ str(AC_setup3_configAAC) + ' Setup 4-'+ str(AC_setup4_configAAC) + ' Setup 5-'+ str(AC_setup5_configAAC))



#Z-score
zScoreSeeds_trainValidation = zScoreNormalization(trainValidation_seeds.tolist())
zScoreSeeds_test = zScoreNormalization(test_seeds.tolist())
#I organize the different configurations for training and validation
#the code here relates to the code I wrote for the second assignments
(trainSeeds, validateSeeds, trainLabels, validateLabels) = generateFolds(zScoreSeeds_trainValidation, trainValidation_labels)

zScore_setup1 = (trainSeeds[0], validateSeeds[0], trainLabels[0], validateLabels[0])
zScore_setup2 = (trainSeeds[1], validateSeeds[1], trainLabels[1], validateLabels[1])
zScore_setup3 = (trainSeeds[2], validateSeeds[2], trainLabels[2], validateLabels[2])
zScore_setup4 = (trainSeeds[3], validateSeeds[3], trainLabels[3], validateLabels[3])
zScore_setup5 = (trainSeeds[4], validateSeeds[4], trainLabels[4], validateLabels[4])



#Plot Generator from dataset
def plotTrainingGenerator(seeds, labels, normalizationMethodLabel): #data will be a list of cars
    f1_1 = []
    f2_1 = []
    f1_2 = []
    f2_2 = []
    f1_3 = []
    f2_3 = []
    i = 0
    while (i < len(seeds)):
        if  labels[i] == 1:
            f1_1.append(seeds[i][0])
            f2_1.append(seeds[i][1])
        elif labels[i] == 2:
            f1_2.append(seeds[i][0])
            f2_2.append(seeds[i][1])
        elif labels[i] == 3:
            f1_3.append(seeds[i][0])
            f2_3.append(seeds[i][1])
        i += 1
        
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(f1_1, f2_1,s=10, c='b', marker="o", label='Kama [Training]')
    ax1.scatter(f1_2,f2_2, s=10, c='r', marker="o", label = 'Rosa [Training]')
    ax1.scatter(f1_3, f2_3,s=10, c='g', marker="o", label = 'Canadian [Training]')
    plt.legend(loc='lower right')
    plt.title('Seeds: Training Dataset [Normalization: '+normalizationMethodLabel+']', fontweight="bold")
    plt.xlabel('Feature 1: Seed Area')
    plt.ylabel('Feature 2: Kernel Width')
    plt.show()