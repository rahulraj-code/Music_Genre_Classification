# Music genre classification basen on their Mel frequency ceptral coefficient


import numpy as np
import librosa
import os
import librosa.display
#import sklearn
import matplotlib.pyplot as plt
#import scipy.io.wavfile as wav

#from tempfile import TemporaryFile
import pickle
import random
import operator
import math



directory = r"D:\tiniminipro\MGC\genres\train"
# creating .dat file to store the features

# loading and extracting features from training set
# covariance => correlation between matrix
# mean_matrix => mean of mfcc_matrix ( mean of column wise)

f = open("my.dat", 'wb')
i = 1
count = 0
for file in os.listdir(r"D:\tiniminipro\MGC\genres\train"):
    (song, sampling_rate) = librosa.load(os.path.join(r"D:\tiniminipro\MGC\genres\train", file))
    mfcc = librosa.feature.mfcc(song, sr=sampling_rate)

    covariance = np.cov(np.matrix.transpose(mfcc))
    mean_matrix = mfcc.mean(0)
    feature = (mean_matrix, covariance, i)
    check = []
    check.append(feature)
    if (check[0][0].shape == (1293,)):
        pickle.dump(feature, f)
    count += 1
    if (count == 99):
        i += 1
        count = 0
    if (i == 10 and count == 99):
        break

f.close()





dataset = []
def loadDataset(filename , split , trainSet , testSet):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

    for x in range(len(dataset)):
        if random.random() <split :
            trainSet.append(dataset[x])
        else:
            testSet.append(dataset[x])

trainingSet = []
testSet = []
# loading dataset and splitting it into training and cross validation set
loadDataset("my.dat" , 0.66, trainingSet, testSet)




# distance between 2 matrices
def distance(instance1 , instance2 , k ):
    distance =0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    # np.trace = sum along diagonal
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 ))
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance

# gives a list of nearest genre till k
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors



# extracting features from the test case or a single sample.
def extract_feature(audio, sampl_rate):
    mfcc = librosa.feature.mfcc(audio, sr=sampl_rate)
    covariance = np.cov(np.matrix.transpose(mfcc))
    mean_matrix = mfcc.mean(0)
    feature = (mean_matrix, covariance, 0)

    return feature


# getting the nearest neighbor (neighbor with maximum count in range of k)
def classify(neighbors):
    genre_count ={}
    for i in neighbors:
        if(i in genre_count):
            genre_count[i]+=1
        else:
            genre_count[i]=1
    prediction = sorted(genre_count.items(), key = operator.itemgetter(1), reverse=True)
    return prediction[0][0]

# gives name to the prediction
def prediction(x):
    if(x==1):
        return "Blues"
    elif(x==2):
        return "Classical"
    elif(x==3):
        return "Country"
    elif(x==4):
        return "Disco"
    elif(x==5):
        return "Hiphop"
    elif(x==6):
        return "Jazz"
    elif(x==7):
        return "Metal"
    elif(x==8):
        return "POp"
    elif(x==9):
        return "Reggae"
    else :
        return "Rock"


def accuracy(predicion, y):
    correct = 0;
    for i in range(len(y)):
        if (prediction[i] == y[i]):
            correct += 1

    accuracy = 100 * (correct / len(y))
    return accuracy


# first load the test case in . wav file and with time of 30 secs
# it should give a mfcc matrix of 1293 x 1293 dimensions
test ,test_sampling_rate = librosa.load("path_way")
test_features = extract_feature(test,test_sampling_rate)
# set value for k
k=11
test_neighbors = getNeighbors(trainingSet ,test_features,k)
pred = prediction((classify(test_neighbors)))

acc = accuracy(pred,test)
print("Model works with accuracy of %d"%acc)
