import os;
import numpy as np;
import glob;
from numpy import linalg as LA;
from paillier import *;
import thread;
from multiprocessing import Pool;
from threading import Thread;
import time;
import ntpath;
import invwishart;
from pkg.global_functions import globalFunction;
from pkg.DPDPCA.DataUser import DataUserImpl;
import decimal;
import proxyAndDataUser;
import math;
import scipy.sparse as sparse;

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
  
    matrix = data[:,1:];
    
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
    for index in np.ndindex(R.shape):
        encR[index] = encrypt(pub,R[index]);
    '''   
    for i in range(0,R.shape[0]):
        for j in range(0,R.shape[1]):
            encR[i,j] = encrypt(pub,R[i,j]);
            
    it = np.nditer(R, flags=['multi_index']);
    while not it.finished:
        #print "%d <%s>" % (it[0], it.multi_index),
        #print str(it.multi_index)+","+str(it[0]);
        encR[it.multi_index] = encrypt(pub,it[0]);
        it.iternext();
    '''
    #print "Encrypt v:";
    encV = np.empty(v.shape[0],dtype=np.dtype(decimal.Decimal));
    for i in range(len(v)):
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
    
def privateLocalPCA(data,k,epsilon):
        
    k = np.minimum(k,LA.matrix_rank(data));
    #print "In each data owner, the k is: %d" % k;

    C = np.dot(data.T,data);

    df = len(C)+1;
    sigma = 1/epsilon*np.identity(len(C));
    #print sigma;
    wishart = invwishart.wishartrand(df,sigma);
    noisyC = C + wishart;
    w, V = sparse.linalg.eigs(noisyC, k=k);
    #U, s, V = LA.svd(C);
    S = np.diagflat(np.sqrt(w));
#    print U[:,0:k].shape;
#    print S[0:k,0:k].shape;
    P = np.dot(V[:,:k],S[:k,:k]);
    #sqrtS = np.sqrt(s);
    #print sqrtS;
    #tmpSum = np.sum(sqrtS);
    #print [elem/tmpSum for elem in sqrtS];
    return P;
        
def privateGlobalPCA(folderPath):
    
    epsilon = 0.5;
    # Get the folder, which contains all the horizontal data.
    dataFileList = glob.glob(folderPath+"/*");
    data = np.loadtxt(dataFileList[0],delimiter=",");
    # Here, it should be OK with data[:,1:data.shape[1]];
    matrix = data[:,1:];
    k = matrix.shape[1]-1;
    #k = 5;
    #print k;
    #dataOwner = DataOwnerImpl(dataFileList[0]);
    P = privateLocalPCA(matrix,k,epsilon);
    #print P.shape;
    dataFileList.pop(0);
    #print "Private Global PCA computing:";

    for path in dataFileList:
        #print str(int(round(time.time() * 1000)))+", "+path;
        #dataOwner = DataOwnerImpl(path);
        tmpData = np.loadtxt(path,delimiter=",");
        PPrime = privateLocalPCA(tmpData[:,1:],k,epsilon);
        
        k_prime = np.maximum(np.minimum(LA.matrix_rank(tmpData[:,1:]),k),LA.matrix_rank(P));
        
        tmpSummary = np.concatenate((PPrime, P), axis=1);
        #P = privateLocalPCA(tmpSummary.T,k_prime,epsilon);
        #print tmpSummary.shape; 
    #print P[:,0];
    return P;
    
def dataOwnerJob(filePath,encFolderPath,pub):
    #print str(int(round(time.time() * 1000)))+", "+filePath+", data share encryption start...";
    encR,encV,encN = calcAndEncryptDataShare(filePath,pub);
    fileName = ntpath.basename(filePath);
    saveShares(encFolderPath,fileName,encR,encV,encN);
    #print str(time.ctime(time.time()))+", "+filePath+", data share encryption done!";
         
def test_myWork(outputFolderPath,encFolderPath,topK):
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
    proxyAndDataUser.dataUserJob(sumEncR,sumEncV,sumEncN,priv,pub,topK);
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
    
    datasets = ['german','madelon','CNAE_2','face2','Amazon_3','Amazon_10'];
    totalRound = 2;
    numDataPerOwner = 3;
    xDataOwners = np.arange(100,1000,100);
    
    for dataset in datasets:
        print "++++++++++++++++ "+ dataset + " ++++++++++++++++++++";
        datasetPath = "input/"+dataset+"_prePCA";
        # 1) Setup the testing environments by creating the horizontal data.
        outputFolderPath = datasetPath+"_referPaper/plaintext/";
        encFolderPath = datasetPath + "_referPaper/ciphertext/";
        if not os.path.exists(encFolderPath):
            os.system('mkdir -p %s' % outputFolderPath);
            os.system('mkdir -p %s' % encFolderPath);

        cprResult = np.zeros((len(xDataOwners),3));
        rawData = np.loadtxt(datasetPath,delimiter=",");
        matrixRank = LA.matrix_rank(rawData[:,1:]);
        for k,numDataOwner in np.ndenumerate(xDataOwners):
            print "Testing with %d data owners." % numDataOwner;
            if os.path.exists(encFolderPath):
                os.system("rm -rf "+outputFolderPath);
                os.system("rm -rf "+encFolderPath);
                os.system('mkdir -p %s' % outputFolderPath);
                os.system('mkdir -p %s' % encFolderPath);

            for j in range(totalRound):
                indices = np.random.randint(rawData.shape[0], size=(numDataOwner, numDataPerOwner))
                for i in range(indices.shape[0]):
                    np.savetxt(outputFolderPath+str(i),rawData[indices[i]],delimiter=",");
                cprResult[k][0] += numDataOwner;
                myWorkTime = test_myWork(outputFolderPath,encFolderPath,matrixRank);
                cprResult[k][1] += myWorkTime;
                print "========================";
                otherWorkTime = test_otherWork(outputFolderPath);
                cprResult[k][2] += otherWorkTime;
                print "------------------------";
                
        for result in cprResult/totalRound:
            print "%d,%d,%d" % (result[0],result[1],result[2]);
            
            
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