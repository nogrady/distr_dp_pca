from pkg.global_functions import globalFunction;
import timeit;
import os;
import sys;
import numpy as np;
from numpy import linalg as LA;
from pkg.paillier import paillierImpl;
import decimal;
import math;

def discretize(b):
    bMin = np.min(b);
    bMax = np.max(b);
    e = 20;
    multi = pow(2,e)-1;
    disB = np.empty(len(b),dtype=np.dtype(decimal.Decimal));
    for i in range(0,len(b)):
        disB[i] = math.floor(multi*np.divide(b[i]-bMin,bMax-bMin));
    return disB;
def preSecurePowerIteration(R,v,N,pub,pri):
    # 2) Encrypt R,v
    encR = np.empty((R.shape[0],R.shape[1]),dtype=np.dtype(decimal.Decimal));
    for i in range(0,R.shape[0]):
        for j in range(0,R.shape[1]):
            encR[i,j] = paillierImpl.encrypt(pub,R[i,j]);
    encV = np.empty(v.shape[0],dtype=np.dtype(decimal.Decimal));
    for i in range(0,len(v)):
        encV[i] = paillierImpl.encrypt(pub,v[i]);
    return encR,encV;
def securePowerIteration(encR,encV,N,b,pub,pri):
    # 3) calculate enc(vTb),enc(Rb);
    #b = np.random.randint(100, size=(len(v), 1));
    #print b;
    #b_times_N = np.multiply(b,N);
    #print b_times_N;
    encvTb = paillierImpl.encrypt(pub,0);
    for i in range(0,len(encV)):
        encvTb = paillierImpl.e_add(pub, encvTb, paillierImpl.e_mul_const(pub, encV[i], b[i]));
    '''
    To verify if the homomorphic addition and multiplication of vTb are correct.
    vTb = 0;
    for i in range(0,len(v)):
        vTb = vTb + v[i]*b[i];
    print vTb;
    '''
    #print encvTb;
    #print decrypt(priv, pub, encvTb);
    encRb = np.empty(b.shape[0],dtype=np.dtype(decimal.Decimal));
    temp = paillierImpl.encrypt(pub,0);
    for i in range(0,encR.shape[0]):
        for j in range(0,encR.shape[1]):
            temp = paillierImpl.e_add(pub, temp, paillierImpl.e_mul_const(pub, encR[i,j], b[j]));
        encRb[i] = temp;
    #print encRb;
    '''
    To verify if the homomorphic addition and multiplication of Rb are correct
    for i in range(0,len(encRb)):
        print decrypt(priv, pub, encRb[i]);
    Rb = np.zeros(len(b));
    temp = 0;
    for i in range(0,R.shape[0]):
        for j in range(0,R.shape[1]):
            temp = temp + R[i,j]*b[j];
        Rb[i] = temp;
    print Rb;
    '''
    # 4) Calculate enc(Rb)-enc(v)^(vTb)/N;
    vTb_dv_N = int(np.divide(paillierImpl.decrypt(priv, pub, encvTb),N)); 
    #print vTb_dv_N;
    encSb = np.empty(b.shape[0],dtype=np.dtype(decimal.Decimal));
    # it is possible that enc(Rb)-enc(v)^(vTb)/N is negative, so...
    signSb = np.empty(b.shape[0],dtype=np.dtype(decimal.Decimal));
    
    for i in range(0,len(encSb)):
        encvvTb_dv_N = paillierImpl.e_mul_const(pub, encV[i], vTb_dv_N);
        if(paillierImpl.decrypt(priv,pub,encRb[i])>=paillierImpl.decrypt(priv,pub,encvvTb_dv_N)):
            vTb_dv_N_neg = pub.n_sq - vTb_dv_N;
            encvvTb_dv_N = paillierImpl.e_mul_const(pub, encV[i], vTb_dv_N_neg);
            encSb[i] = paillierImpl.e_add(pub, encRb[i], encvvTb_dv_N);
            signSb[i] = 1.0;
        else:
            encSb[i] = paillierImpl.e_add(pub, (pub.n_sq - encRb[i]), encvvTb_dv_N);
            signSb[i] = -1.0;
    #print encSb;
    Sb = np.empty(b.shape[0],dtype=np.dtype(decimal.Decimal));
    Sb_temp = 0;
    
    # Decrypt Sb and calculate the norm of Sb
    for i in range(0,len(Sb)):
        Sb[i] = paillierImpl.decrypt(priv, pub, encSb[i])*signSb[i];
        Sb_temp = Sb_temp + Sb[i]*Sb[i];
    Sb_norm = math.sqrt(Sb_temp);
    #print Sb_norm;
    for i in range(0,len(Sb)):
        Sb[i] = np.divide(Sb[i],Sb_norm);
    #print Sb;
    #print "=========================";
    '''
    The naive way to discretize the normalized Sb.
    SbMax = np.max(Sb);
    SbMin = np.min(Sb);
    for i in range(0,len(Sb)):
        Sb[i] = Sb[i]*np.divide(SbMax,SbMin);
        #print Sb[i];
    Sb_int = np.rint(Sb.astype(np.double));
    #print Sb_int;
    '''
    Sb_int = discretize(Sb);
    '''
    To verify if Sb are correctly calculated under encryption.
    for i in range(0,len(encSb)):
        print decrypt(priv, pub, encSb[i]);
    print "--------------------------------";
    Rb = np.empty(b.shape[0],dtype=np.dtype(decimal.Decimal));
    for i in range(0,len(encRb)):
        Rb[i] = decrypt(priv, pub, encRb[i]);
        print Rb[i]-v[i]*vTb_dv_N;
    '''
    return Sb_int,Sb;
def prePowerIteration(R,v,N):
    vTv = np.zeros((len(v),len(v)));
    for i in range(0,len(v)):
        for j in range(0,len(v)):
            vTv[i,j] = v[i]*v[j];
    
    vTv_divide_N = np.divide(vTv,N);        
    S = R - vTv_divide_N;
    return S;

def powerIteration(S,b):
    Sb = np.dot(S,b);
    Sb = np.divide(Sb,LA.norm(Sb));
    return Sb;


inputFilePath = "./input/madelon_prePCA_training";
# 1) Load the data from text file, then calculate R,v,N, here, to simplify the experiment process, 
# I just assume only one data owner, or just ignore the aggregation process.  
data = np.loadtxt(inputFilePath,delimiter=",");
# Here, it should be OK with data[:,1:data.shape[1]];
matrix = data[:,range(1,data.shape[1])];

if not isinstance(matrix,(int, long )):
    matrix = matrix.astype(int);

#print matrix.T.shape;
#print matrix.shape;
#matrix = np.random.randint(100,size=(5,6));
v = np.sum(matrix,axis=0);
N = matrix.shape[0];

R = np.dot(matrix.T,matrix);

priv, pub = paillierImpl.generate_keypair(128);

S = prePowerIteration(R,v,N);
encR,encV = preSecurePowerIteration(R,v,N,pub,priv);

for j in range(0,10):
    b = np.random.randint(100, size=(len(v), 1));
    b = np.multiply(b,N);
    Sb = b;
    epsolon = 0.01;
    i= 0;
    tempSbNorm = 1;
    tempbNorm = 1;
    lastBNorm = Sb;
    start = timeit.default_timer();
    while tempbNorm > epsolon:
        #print str(i)+"th iteration:";
        bTemp,b_norm = securePowerIteration(encR,encV,N,b.astype(int),pub,priv);
        #SbTemp = powerIteration(S,Sb);
        #tempSbNorm = LA.norm(Sb-SbTemp);
        tempbNorm = LA.norm(b_norm-lastBNorm);
        #print tempSbNorm;
        #print tempbNorm;
        #print b_norm;
        print "==========================";
        #Sb = SbTemp;
        b = bTemp;
        lastBNorm = b_norm;
        i = i+1;
    end = timeit.default_timer();
    print i;
    print (end-start);