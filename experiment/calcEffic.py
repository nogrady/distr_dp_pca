import numpy as np;

resultPath = "../exper_results/efficiency/german/rawResult2";
data = np.loadtxt(resultPath,delimiter=",",dtype = str);
while data.size != 0:
    a = (int(data[2]) - int(data[1]))*1.0/int(data[0]);
    b = int(data[6])-int(data[3]);
    c = int(data[9])-int(data[8]);
    print str(a+b)+","+str(c);
    data = np.delete(data,range(0,11),axis=0);
    
