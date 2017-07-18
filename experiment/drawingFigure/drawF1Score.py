import numpy as np;
import matplotlib.pyplot as plt;

resultPath = "../exper_results/f1Score/madelon/result.csv";
data = np.loadtxt(resultPath,delimiter=",");

#x = range(1,len(data)+1);
x = [10,40,70,100,130,160,190,220,250,280,310,340];
y1Line,y2Line = plt.plot(x, data[:,0], 'bo-', x, data[:,1], 'r^-');
plt.legend([y2Line,y1Line], ['DPDPCA', 'PCA'],loc=2);
plt.axis([0,350,0.5,1]);
#plt.axis([0,10,0.4,1.0]);
plt.xlabel('Number of Principal Components',fontsize=18);
plt.ylabel('F1-Score',fontsize=18);
plt.title('Madelon Synthetic Dataset', fontsize=18);
plt.xticks(x);
#plt.show();

plt.savefig('../exper_results/f1Score_madelon.pdf', format='pdf', dpi=1000);