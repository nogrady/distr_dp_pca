import os;
import numpy as np;
import glob;
from pkg.DPDPCA.DataOwner import DataOwnerImpl;
import dataOwnerShare;
from numpy import linalg as LA;
from paillier import *;
import thread;
from multiprocessing import Pool;
from threading import Thread;
import time;
import ntpath;
from pkg.global_functions import globalFunction;
from pkg.DPDPCA.DataUser import DataUserImpl;
import time;
import decimal;
import proxyAndDataUser;

'''
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
'''

'''
How to use Multi-thread (not multi-process) in Python.
'''
def callDataOwners(folderPath,encFolderPath,pub):
        try:
            dataFileList = glob.glob(folderPath+"/*");
            for path in dataFileList:
                t = Thread(target=dataOwnerJob, args=(path,encFolderPath,pub,));
                t.start();
        except:
            print "Error: unable to start thread"
        return;
    
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

def saveEncrypedData(outputPath,matrix):
    np.savetxt(outputPath,matrix,delimiter=",",fmt="%0x");
    
def saveShares(encFolderPath,fileName,encR,encV,encN):
    
    encRFilePath = encFolderPath+"encR/"+fileName;
    encVFilePath = encFolderPath+"encV/"+fileName;
    encNFilePath = encFolderPath+"encN/"+fileName;
    
    if not os.path.exists(encFolderPath+"encR/"):
        os.mkdir(encFolderPath+"encR/");
        os.mkdir(encFolderPath+"encV/");
        os.mkdir(encFolderPath+"encN/");
    
    saveEncrypedData(encRFilePath,encR);
    saveEncrypedData(encVFilePath,encV);
    saveEncrypedData(encNFilePath,encN);
        
def privateGlobalPCA(folderPath):
    
    epsilon = 0.5;
    # Get the folder, which contains all the horizontal data.
    dataFileList = glob.glob(folderPath+"/*");
    data = np.loadtxt(dataFileList[0],delimiter=",");
    # Here, it should be OK with data[:,1:data.shape[1]];
    matrix = data[:,1:data.shape[1]];
    k = matrix.shape[1]-1;
    #k = 5;
    #print k;
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
    
def dataOwnerJob(filePath,encFolderPath,pub):
    #print str(int(round(time.time() * 1000)))+", "+filePath+", data share encryption start...";
    encR,encV,encN = calcAndEncryptDataShare(filePath,pub);
    fileName = ntpath.basename(filePath);
    saveShares(encFolderPath,fileName,encR,encV,encN);
    #print str(time.ctime(time.time()))+", "+filePath+", data share encryption done!";
         
def test_myWork(outputFolderPath,encFolderPath):
    priv, pub = generate_keypair(128);
    dataFileList = glob.glob(outputFolderPath+"*");
        # 2.1) DataOwnerJob
    startTime = int(round(time.time() * 1000));
    print str(startTime)+", Data Owners encrypt starts..";
    for path in dataFileList:
        dataOwnerJob(path,encFolderPath,pub);
        # 2.2) Proxy Job
    endTime = int(round(time.time() * 1000));
    print str(endTime)+", Data Owners encrypt ends..";    
    dataOwnerTime = (endTime-startTime)/len(dataFileList);
    startTime = int(round(time.time() * 1000));
    sumEncR,sumEncV,sumEncN = proxyAndDataUser.paraAggrtEncryptedData(encFolderPath,pub);
#    encRList = glob.glob(encFolderPath+"encR/*");
#    sumEncR = paraAggregate(proxy,encRList);
    #sumEncR,sumEncV,sumEncN = proxyAndDataUser.aggrtEncryptedDataOwnerShare(encFolderPath,pub);
        # 2.3) Data User Job
    proxyAndDataUser.dataUserJob(sumEncR,sumEncV,sumEncN,priv,pub);
    endTime = int(round(time.time() * 1000));
    totalTime = dataOwnerTime + (endTime-startTime);
    return totalTime;

def test_otherWork(dataSetPath):
    startTime = int(round(time.time() * 1000));
    print str(startTime)+", Private Global PCA starts..";
    privateGlobalPCA(dataSetPath);
    endTime = int(round(time.time() * 1000));
    print str(endTime)+", Private Global PCA ends..";
    return (endTime-startTime); 
#validateEncScheme();

if __name__ == "__main__":
    
    datasets = ['diabetes','australian','german', 'ionosphere', 'madelon'];
    
    for dataset in datasets:
        print "++++++++++++++++ "+ dataset + " ++++++++++++++++++++";
        dataSetPath = "input/"+dataset+"_prePCA";
        # 1) Setup the testing environments by creating the horizontal data.
        outputFolderPath = dataSetPath+"_referPaper/plaintext/";
        encFolderPath = dataSetPath + "_referPaper/ciphertext/";
        
        cprResult = np.zeros((10,3));
        totalRound = 5;
        for j in range(0,totalRound):
            if not os.path.exists(encFolderPath):
                    os.system('mkdir -p %s' % outputFolderPath);
                    os.system('mkdir -p %s' % encFolderPath);
            
            numOfTrunks = 10;
            for i in range(0,10):
                
                numOfTrunks = 10 + i*20;
                cprResult[i][0] = cprResult[i][0]+numOfTrunks;
                globalFunction.splitAndSaveDatasets(dataSetPath,outputFolderPath,numOfTrunks);             
                #==================================================
                print numOfTrunks;
                myWorkTime = test_myWork(outputFolderPath,encFolderPath);
                cprResult[i][1] = cprResult[i][1]+myWorkTime;
                print "========================";
                otherWorkTime = test_otherWork(outputFolderPath);
                cprResult[i][2] = cprResult[i][2]+otherWorkTime;
                print "------------------------";
                #==================================================
            os.system("rm -rf "+outputFolderPath);
            os.system("rm -rf "+encFolderPath);
            
        for i in range(0,len(cprResult)):
                print "%d, %d, %d" % (cprResult[i][0]/totalRound,cprResult[i][1]/totalRound,cprResult[i][2]/totalRound);
            
            
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