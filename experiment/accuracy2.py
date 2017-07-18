from pkg.global_functions import globalFunction as gf;
from sklearn.model_selection import StratifiedKFold;
from sklearn.model_selection import GridSearchCV;

import timeit;
import os;
import sys;
from pkg.DPDPCA.DataOwner import DataOwnerImpl;
import numpy as np;
import glob;
from numpy import linalg as LA;
from pkg.wishart import invwishart;
from sklearn import svm;

def genTrainingTestingData(dataPath,trainingFilePath,testingFilePath):
    data = np.loadtxt(dataPath,delimiter=",");
    #print data.shape[0];
    shuffleData = np.random.permutation(data);
    testIndex = -shuffleData.shape[0]/3;
    testMatrix = shuffleData[testIndex:-1,:];
    np.savetxt(testingFilePath,testMatrix,delimiter=",",fmt='%1.7f');
    numOfPositive = 0;
    numOfNegative = 0
    for i in range(0,testMatrix.shape[0]):
        if testMatrix[i,0]>0:
            numOfPositive = numOfPositive + 1;
        else:
            numOfNegative = numOfNegative + 1;
    print "Number of testing samples: "+ str(testMatrix.shape[0]);
    print "Number of positive samples: " + str(numOfPositive);
    print "Number of negative samples: " + str(numOfNegative);
    #print testMatrix.shape[0];
    #print testMatrix;
    trainMatrix = shuffleData[:(shuffleData.shape[0]+testIndex),:];
    
    numOfPositive = 0;
    numOfNegative = 0
    for i in range(0,trainMatrix.shape[0]):
        if trainMatrix[i,0]>0:
            numOfPositive = numOfPositive + 1;
        else:
            numOfNegative = numOfNegative + 1;
    print "Number of training samples: "+ str(trainMatrix.shape[0]);
    print "Number of positive samples: " + str(numOfPositive);
    print "Number of negative samples: " + str(numOfNegative);
    
    '''
    Make the training data column-centered.
    '''
    pureData = trainMatrix[:,1:];
    columnMean = np.mean(pureData,axis=0);
    centeredTrainMatrix = pureData - columnMean;
    label = trainMatrix[:,0];
    label = label[:,np.newaxis];
    labeledCenteredTrainingMatrix = np.concatenate((label,centeredTrainMatrix),axis=1);
    np.savetxt(trainingFilePath,labeledCenteredTrainingMatrix,delimiter=",",fmt='%1.7f');
    #print trainMatrix.shape[0];
    return columnMean;

def privateGlobalPCA(folderPath,k):
    
    epsilon = 0.6;
    # Get the folder, which contains all the horizontal data.
    dataFileList = glob.glob(folderPath+"/*");
    data = np.loadtxt(dataFileList[0],delimiter=",");
    # Here, it should be OK with data[:,1:data.shape[1]];
    matrix = data[:,range(1,data.shape[1])];
    #k = matrix.shape[1]-1;
    #k = 5;
    print k;
    dataOwner = DataOwnerImpl(dataFileList[0]);
    P = dataOwner.privateLocalPCA(None,k,epsilon);
    #print P.shape;
    dataFileList.pop(0);
    #print "Private Global PCA computing:";

    for path in dataFileList:
        #print str(int(round(time.time() * 1000)))+", "+path;
        dataOwner = DataOwnerImpl(path);
        PPrime = dataOwner.privateLocalPCA(None,k,epsilon);
        
        k_prime = np.maximum(np.minimum(LA.matrix_rank(dataOwner.data),k),LA.matrix_rank(P));
        
        tmpSummary = np.concatenate((PPrime, P), axis=1);
        
        P = dataOwner.privateLocalPCA(tmpSummary.T,k_prime,epsilon);
        #print tmpSummary.shape; 

    #print P[:,0];
    return P;
    
def myGlobalPCA(folderPath):
    dataFileList = glob.glob(folderPath+"/*");
    data = np.loadtxt(dataFileList[0],delimiter=",");
    sumR = None;
    sumV = None;
    sumN = 0;
    for path in dataFileList:
        dataOwner = DataOwnerImpl(path);
        R = np.dot(dataOwner.data.T,dataOwner.data);       
        v = np.sum(dataOwner.data,axis=0);
        N = dataOwner.data.shape[0];
        if sumR is None:
            sumR = R;
            sumV = v;
        else:
            sumR = sumR + R;
            sumV = sumV + v;
            sumN = sumN + N;
            
    S = sumR - np.divide(np.outer(v,v),sumN);
    
    U, s, V = np.linalg.svd(S);
    S = np.diagflat(np.sqrt(s));
    
    #print U.dot(S)[:,0];
    return U.dot(S);

def svmRBFKernel(trainingData,trainingLabel,testingData,testingLabel):
    skfCV = StratifiedKFold(n_splits=10,shuffle=True);
        
    # 2). Grid search, with the C and gamma parameters.
    C_range = np.logspace(-3, 3, 8);
    gamma_range = np.logspace(-3, 3, 8);
    param_grid = dict(gamma=gamma_range, C=C_range);
    # Notice here that the svm.SVC is just for searching for the parameter, we didn't really train the model yet.  
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, n_jobs =2, cv=skfCV);
    #grid.fit(ldaProjTrainingData, trainingLabel);
    grid.fit(trainingData, trainingLabel);
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_));
    
    # 3). Train the SVM model using the parameters from grid search. 
    svmClf = svm.SVC(C = grid.best_params_['C'], gamma = grid.best_params_['gamma'], kernel='rbf');
    svmClf.fit(trainingData,trainingLabel);
    #print svmClf.support_;
    # 4). Predict the testing data and calculate the f1 score.
    svmPred = svmClf.predict(testingData);
    result = gf.calcF1Score(testingLabel,svmPred);
    return result;
if __name__ == "__main__":
    
    dataSetPath = "input/diabetes_prePCA";
    outputFolderPath = dataSetPath+"_referPaper2/plaintext/";
    trainingDataPath = "input/diabetes_prePCA_training";
    testingDataPath = "input/diabetes_prePCA_testing";
    columnMean = genTrainingTestingData(dataSetPath,trainingDataPath,testingDataPath);
    numOfTrunks = 20;
    gf.splitAndSaveDatasets(trainingDataPath,outputFolderPath,numOfTrunks);
    #pgProjMatrix = privateGlobalPCA(outputFolderPath);
    #projMatrix = myGlobalPCA(outputFolderPath);
    
    trainingData = np.loadtxt(trainingDataPath,delimiter=",");
    pureTrainingData = trainingData[:,1:];
    trainingLabel = trainingData[:,0];
    
    testingData = np.loadtxt(testingDataPath,delimiter=",");
    pureTestingData = testingData[:,1:];
    testingLabel = testingData[:,0];
    centeredTestingData = pureTestingData - columnMean;
    projMatrix = myGlobalPCA(outputFolderPath);
    pgResultList=[];
    myResultList=[];
    
    for k in range(1,6):
        pgProjMatrix = privateGlobalPCA(outputFolderPath,k); 
        #print pgProjMatrix.shape;   
        projTrainingData = np.dot(pureTrainingData,pgProjMatrix);
        projTestingData = np.dot(centeredTestingData,pgProjMatrix);
        result = svmRBFKernel(projTrainingData,trainingLabel,projTestingData,testingLabel);
        pgResultList.append(result);
        
        kProjMatrix = projMatrix[:,0:k];
        projTrainingData = np.dot(pureTrainingData,kProjMatrix);
        projTestingData = np.dot(centeredTestingData,kProjMatrix);
        result = svmRBFKernel(projTrainingData,trainingLabel,projTestingData,testingLabel);
        myResultList.append(result);
        print "===========================";
        #return result;
    for i in range(0,len(pgResultList)):
        print "%f , %f" % (pgResultList[i][2],myResultList[i][2]);