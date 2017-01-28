import os;
import sys;
import copy;

f1scorefile = str(sys.argv[1]);
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
