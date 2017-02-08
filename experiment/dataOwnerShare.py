import os;
import numpy as np;
from paillier import *;
import math;
import decimal;

def calcAndEncryptDataShare(dataSharePath,pub):
    data = np.loadtxt(dataSharePath,delimiter=",");
    # Here, it should be OK with data[:,1:data.shape[1]];
    matrix = data[:,range(1,data.shape[1])];
    
    if not isinstance(matrix,(int, long )):
        matrix = matrix.astype(int);
           
    R = np.dot(matrix.T,matrix);       
    v = np.sum(matrix,axis=0);
    N = data.shape[0];
    #matrix = discretize(matrix);
    
    #print "Plaintext R:";
    #print R;
    
    #print R.shape;
    #print v.shape;
    #print N;
    #R = R.astype(int);
    #v = np.floor(v);
    #print "Encrypting R:";
    encR = np.empty((R.shape[0],R.shape[1]),dtype=np.dtype(decimal.Decimal));
    for i in range(0,R.shape[0]):
        for j in range(0,R.shape[1]):
            encR[i,j] = encrypt(pub,R[i,j]);
    '''        
    it = np.nditer(R, flags=['multi_index']);
    while not it.finished:
        #print "%d <%s>" % (it[0], it.multi_index),
        #print str(it.multi_index)+","+str(it[0]);
        encR[it.multi_index] = encrypt(pub,it[0]);
        it.iternext();
    '''
    #print "Encrypt v:";
    encV = np.empty(v.shape[0],dtype=np.dtype(decimal.Decimal));
    for i in range(0,len(v)):
        encV[i] =  encrypt(pub,v[i]);
    '''
    it = np.nditer(v, flags=['multi_index'])
    while not it.finished:
        #print "%d <%s>" % (it[0], it.multi_index),
        encV[it.multi_index] = encrypt(pub,it[0]);
        it.iternext();
    '''
    encN = encrypt(pub,N);
    return encR,encV,(encN,);

def discretize(R):
    #e is the number of bits, not the natural number.
    e = 8;
    cMax = np.amax(R,axis=0);
    #print cMax;
    cMin = np.amin(R,axis=0);
    #print cMin;
    intR = np.floor(np.divide((math.pow(2,e)-1)*(R - cMin),(cMax-cMin)));
    return intR;

def saveEncrypedData(outputPath,matrix):
    np.savetxt(outputPath,matrix,delimiter=",",fmt="%0x");
    
def saveShares(encFolderPath,fileName,encR,encV,encN):
    
    encRFilePath = encFolderPath+"encR/"+fileName;
    encVFilePath = encFolderPath+"encV/"+fileName;
    encNFilePath = encFolderPath+"encN/"+fileName;
    saveEncrypedData(encRFilePath,encR);
    saveEncrypedData(encVFilePath,encV);
    saveEncrypedData(encNFilePath,encN);
    
    '''
    np.savetxt(encRFilePath,encR,delimiter=",");
    np.savetxt(encVFilePath,encV,delimiter=",");
    np.savetxt(encNFilePath,np.array(encN).reshape(1,),delimiter=",");
    '''
'''    
dataSharePath = "./input/australian_prePCA_referPaper/0";
R,v,N = calcDataShare(dataSharePath);
priv, pub = generate_keypair(128);
print "public key is "+ str(pub);
encR,encV,encN = encryptDataShare(R,v,N,pub);
encFolderPath = dataSharePath+"_enc/";
saveShares(encFolderPath,encR,encV,encN);
'''

