import timeit;
import os;
import numpy as np;
from numpy import linalg as LA;
import privateLocalPCA as plPCA;
import global_functions as gf;
import glob;
import invwishart;
import time;

def simulate(folderPath):

    # Get the folder, which contains all the horizontal data.
    dataFileList = glob.glob(folderPath+"/*");
    data = np.loadtxt(dataFileList[0],delimiter=",");
    # Here, it should be OK with data[:,1:data.shape[1]];
    matrix = data[:,range(1,data.shape[1])];
    k = matrix.shape[1];
    #k = 4;
    P,rank = plPCA.privateLocalPCA(dataFileList[0],k);
    #print P.shape;
    dataFileList.pop(0);
    #print "Private Global PCA computing:";
    
    for path in dataFileList:
        #print str(int(round(time.time() * 1000)))+", "+path;
        
        PPrime,rank = plPCA.privateLocalPCA(path,k);
        rank = LA.matrix_rank(PPrime);
        
        tmpSummary = np.concatenate((PPrime, P), axis=1);
        
        #print tmpSummary.shape;
        
        C = np.dot(tmpSummary,tmpSummary.T);
        df = len(C)+1;
        sigma = 1/0.6*np.identity(len(C));
        #print sigma;
        wishart = invwishart.wishartrand(df,sigma);
        
        U, s, V = np.linalg.svd(C+wishart, full_matrices=True); 
    #    S = np.diagflat(s);
    #    P = np.dot(U[:,0:k-1],S[0:k-1,0:k-1]);
    #    plPCA.privateLocalPCA(path,k);
        
    # return P;
