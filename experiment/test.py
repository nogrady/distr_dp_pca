import numpy as np;
import scipy as sp;
from numpy import linalg as LA;
import invwishart;
import global_functions as gf;
import os;
import copy;

def distance(v1, v2):
    v = v1 - v2;
    v = v * v;
    return np.sum(v);

def getApproxEigval(covMatrix,r1):
    temp1 = np.dot(covMatrix,r1);
    v1 = np.dot(r1.T,temp1);
    v2 = np.dot(r1.T,r1);
    eigVal = np.divide(v1,v2);
    return eigVal;

def genEigenvectors_power(covMatrix):
#    r0 = np.random.rand(covMatrix.shape[0],1);
    epsilon = 0.01;
    
    k=0;
    while k<covMatrix.shape[0]:
        r0 = np.random.rand(covMatrix.shape[0],1);
        count=0;
        while True:
            r1 = np.dot(covMatrix, r0);
            # Get the second norm of r1;
            scale = LA.norm(r1,2);
            r1 = np.divide(r1,scale);
            #dist = LA.norm(r1-r0,2);
            eigVal = getApproxEigval(covMatrix,r1);
            dist = LA.norm(np.dot(covMatrix,r1)-eigVal*r1,2);
            #print dist;
            
            if dist < epsilon:
                #print count;
                #print eigVal;
                print r1;
                break;
            else:    
                r0 = r1;
                count = count + 1;
        #print (r1.dot(r1.T)); 
        covMatrix = covMatrix - covMatrix.dot(r1.dot(r1.T));
        k = k+1;            
    return r1;

def genNoisyEigenvectors_power(covMatrix,eps):
    epsPrime = np.divide(eps,np.sqrt(4*covMatrix.shape[0]));
    k=0;
    while k<covMatrix.shape[0]:
        x1 = genSingleNoisyEigenvector_power(covMatrix,epsPrime);
        print x1;
        lap = np.random.laplace(0, np.divide(1.0,epsPrime), 1);
        estSigma = LA.norm(np.dot(covMatrix,x1),2)+lap;
        covMatrix = covMatrix - estSigma*np.dot(x1,x1.T);
        k = k+1;

def genSingleNoisyEigenvector_power(covMatrix,epsDiff):
    
    epsilon = 0.01;
    sigma = 2*np.divide(1.0,epsDiff)*np.sqrt(4*covMatrix.shape[0]);

#        r0 = np.random.rand(covMatrix.shape[0],1);
    x0 = np.random.normal(0, np.divide(1.0,covMatrix.shape[0]), covMatrix.shape[0]);
    count = 0;
    while True:
        gt = np.random.normal(0, np.square(sigma), covMatrix.shape[0]);
        x1 = np.dot(covMatrix, x0);
        x1 = x1+gt;
        # Get the second norm of r1;
        scale = LA.norm(x1,2);
        x1 = np.divide(x1,scale);
        #dist = LA.norm(r1-r0,2);
        eigVal = getApproxEigval(covMatrix,x1);
        dist = LA.norm(np.dot(covMatrix,x1)-eigVal*x1,2);
        #print dist;
        if dist < epsilon:
            #print count;
            #print eigVal;
            break;
        else:
            x0 = x1;
            count = count + 1;            
    return x1;
'''
matrix = np.array([[170,0],[175,0],[173,0],[180,0]]);
neighMatrix = np.array([[170,0],[175,0],[173,0],[180,0],[160,5]]);

covMatrix = np.dot(matrix.T,matrix);
w,v = LA.eig(covMatrix);
print v;

print "************";

neiCovMatrix = np.dot(neighMatrix.T,neighMatrix);
w,v = LA.eig(neiCovMatrix);
print v;
'''


testFile = "./input/german_testing_reduced_features";
predictedFile = "./german_testing_reduced_features_formatted.predict";
testFile = "./testFile";
predictedFile = "./predictedFile";
#f1 = gf.calcF1Score(testFile,predictedFile);
#print f1;

f1scorefile = "./f1score_german";
with open(f1scorefile) as f:
    content = f.readlines();

print len(content);
matrix = [];
row = [];
for i in range(0,len(content)):
    
    if content[i].rstrip():
        tmp = content[i].rstrip();
        row.append(tmp.split(',')[1]);
    else:
        matrix.append(copy.copy(row));
        del row[:];


for i in range(0,len(matrix[0])):
    tmpSum = 0;
    for j in range(0,len(matrix)):
        tmpSum = tmpSum + float(matrix[j][i]);
    print str(tmpSum/len(matrix))+',';

'''
matrix = np.array([[2,-12],[1,-5]]);

eigenvector = genEigenvectors_power(matrix);
#print eigenvector;
print "****";

epsDiff = 0.01;
genNoisyEigenvectors_power(matrix,epsDiff);
#noisyEigvector = genSingleNoisyEigenvector_power(matrix,epsDiff);
#print noisyEigvector;
'''
'''
w, v = LA.eig(matrix);
idx = np.absolute(w).argsort()[::-1];
#print idx;
sortedW = w[idx];
#print sortedW;
sortedV = v[:,idx];
#print w;
#print v;
print sortedW;
print sortedV;
print "**********";

U, s, V = LA.svd(matrix, full_matrices=True)
print U;
print LA.norm(U,np.inf);
print V;
print LA.norm(V,np.inf);
'''

