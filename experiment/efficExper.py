import os;
import numpy as np;
import glob;
import dataOwnerShare;
from numpy import linalg as LA;
import copy;
from paillier import *;
import thread;
from threading import Thread;
import time;
import ntpath;
import global_functions as gf;
import proxyAndDataUser;
import time;
import decimal;
import privateGlobalPCA;

def validateEncScheme():
    x = np.arange(20).reshape(4,5);
    testOutputFile = "./input/testEncryption";
    np.savetxt(testOutputFile,x,delimiter=",",fmt='%1d');
    plaintextFolder = testOutputFile+"Folder/plaintext/";
    if not os.path.exists(plaintextFolder):
        os.mkdir(plaintextFolder);
    numOfTrunks = 2;
    gf.splitAndSaveDatasets(testOutputFile,plaintextFolder,numOfTrunks);
    
    encFolderPath = testOutputFile+"Folder/ciphertext/";
    if not os.path.exists(encFolderPath):
        os.mkdir(encFolderPath);
    encRFolderPath,encVFolderPath,encNFolderPath = proxyAndDataUser.initFolders(encFolderPath);
    priv, pub = generate_keypair(128);

    proxyAndDataUser.callDataOwners(plaintextFolder,encFolderPath,pub);
    time.sleep(5);
    sumEncR, sumEncV, sumEncN = proxyAndDataUser.aggrtEncryptedDataOwnerShare(encRFolderPath,encVFolderPath,encNFolderPath,pub);

    v = proxyAndDataUser.dataUserJob(copy.copy(sumEncR),copy.copy(sumEncV),copy.copy(sumEncN),priv,pub);

def test_myWork(outputFolderPath):
    priv, pub = generate_keypair(128);
    dataFileList = glob.glob(outputFolderPath+"*");
        # 2.1) DataOwnerJob
    proxyAndDataUser.initFolders(encFolderPath);
    print str(int(round(time.time() * 1000)))+", Data Owners encrypt starts..";
    for path in dataFileList:
        proxyAndDataUser.dataOwnerJob(path,encFolderPath,pub);
        # 2.2) Proxy Job
    print str(int(round(time.time() * 1000)))+", Data Owners encrypt ends..";    
    sumEncR,sumEncV,sumEncN = proxyAndDataUser.paraAggrtEncryptedData(encFolderPath,pub);
    #sumEncR,sumEncV,sumEncN = proxyAndDataUser.aggrtEncryptedDataOwnerShare(encFolderPath,pub);
        # 2.3) Data User Job
    proxyAndDataUser.dataUserJob(sumEncR,sumEncV,sumEncN,priv,pub);
def test_otherWork(dataSetPath):
    print str(int(round(time.time() * 1000)))+", Private Global PCA starts..";
    privateGlobalPCA.simulate(dataSetPath);
    print str(int(round(time.time() * 1000)))+", Private Global PCA ends..";
    
#validateEncScheme();

dataSetPath = "./input/australian_prePCA";
# 1) Setup the testing environments by creating the horizontal data.
outputFolderPath = dataSetPath+"_referPaper/plaintext/";
encFolderPath = dataSetPath + "_referPaper/ciphertext/";

for j in range(0,10):
    if not os.path.exists(encFolderPath):
            os.mkdir(outputFolderPath);
            os.mkdir(encFolderPath);
    
    numOfTrunks = 10;
    for i in range(0,10):
        numOfTrunks = 10 + i*20;
        gf.splitAndSaveDatasets(dataSetPath,outputFolderPath,numOfTrunks);
         
        #==================================================
        print numOfTrunks;    
        test_myWork(outputFolderPath);
        print "========================";
        test_otherWork(outputFolderPath);
        print "------------------------"
        #==================================================
    os.system("rm -rf "+outputFolderPath);
    os.system("rm -rf "+encFolderPath);

#Test Paillier encryption
'''
priv, pub = generate_keypair(128);
matrix = np.empty((2,2),dtype=np.dtype(decimal.Decimal));
print int(round(time.time() * 1000));
matrix[0,0] = encrypt(pub,257);
print int(round(time.time() * 1000));
matrix[0,1] = encrypt(pub,2);
matrix[1,0] = e_add(pub, matrix[0,0], matrix[0,1]);
matrix[1,1] = decrypt(priv,pub,matrix[1,0]);

print matrix;
'''
"""
matrix[0,1] = x;

print "\n";
np.savetxt("testFile",(x,),delimiter=",",fmt="%0x");
encX = np.loadtxt("testFile",delimiter=",",dtype='str');
print encX;
strX = np.array_str(encX);

print int(strX,16);
y = decrypt(priv,pub,matrix[0,1]);
print y;
"""