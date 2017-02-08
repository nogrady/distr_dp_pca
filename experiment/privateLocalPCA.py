import timeit;
import os;
import numpy as np;
from numpy import linalg as LA;
import invwishart;

def privateLocalPCA(dataSharePath,k):

    data = np.loadtxt(dataSharePath,delimiter=",");
    #print data.shape[0];
    matrix = data[:,range(1,data.shape[1])];
    
    #k = matrix.shape[1];
    
    #Preprocessing, scaling data.
    #cMax = np.amax(matrix,axis=0);
    #cMin = np.amin(matrix,axis=0);
    #matrix = np.divide((matrix - cMin),(cMax-cMin));
    #columnMean = np.mean(matrix,axis=0);
    #for k in range(len(matrix)):
    #    matrix[k,:]=matrix[k,:]-columnMean;
    #covMatrix = np.cov(matrix.T);
    
    C = np.dot(matrix.T,matrix);
    df = len(C)+1;
    sigma = 1/0.6*np.identity(len(C));
    #print sigma;
    wishart = invwishart.wishartrand(df,sigma);

    U, s, V = np.linalg.svd(C+wishart, full_matrices=True);
    rank = LA.matrix_rank(C);
    k = min(k,rank);
    
    S = np.diagflat(s);
#    print U[:,0:k].shape;
#    print S[0:k,0:k].shape;
    P = np.dot(U[:,0:k],S[0:k,0:k]);
    
    return P,rank;
