import numpy as np;
import matplotlib.pyplot as plt;

resultPath = "../exper_results/efficiency/australian/rawResult7";
data = np.loadtxt(resultPath,delimiter=",",dtype = str);
x = [];
y1 = [];
y2 = [];
while data.size != 0:
    a = (int(data[2]) - int(data[1]))*1.0/int(data[0]);
    x.append(int(data[0]));
    b = int(data[6])-int(data[3]);
    y1.append(a+b);
    c = int(data[9])-int(data[8]);
    y2.append(c);
    print str(a+b)+","+str(c);
    data = np.delete(data,range(0,11),axis=0);
 
'''
For the y axis, convert millisecond to second.
'''    
#y1 = [i / 1000 for i in y1];
#y2 = [i / 1000 for i in y2];   
 
'''
Drawing the efficiency figures.
'''
    
y1Line,y2Line = plt.plot(x, y1, 'r^-', x, y2, 'bo-');
plt.legend([y1Line,y2Line], ['DPDPCA', 'Private Local/Global PCA'],loc=2);
plt.axis([0,200,0,600]);
plt.xlabel('Number of Data Owners',fontsize=18);
plt.ylabel('Time (ms)',fontsize=18);
#plt.title('German Credit Dataset',fontname="Times New Roman Bold");
plt.title('Australian Credit Dataset (460 samples)',fontsize=18);
#yticks = range(100,1100,100);
plt.xticks(x);
#plt.yticks(yticks);
#plt.show();

plt.savefig('../exper_results/effic_australian.pdf', format='pdf', dpi=1000);