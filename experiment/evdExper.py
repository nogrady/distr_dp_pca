from pkg.global_functions import globalFunction as gf;
import timeit;
import os;
import sys;

'''
1) Split data samples into training samples and testing samples
2) Generating Eigenvectors (normal vs noisy[dp]) from the training samples, get the principal components (projection matrix)
3) Project the testing samples.
4) SVM classification, dimension reduced training samples, dimension reduced testing samples.
'''

def pca_svm_classification(cMin, cMax, eigenvectors, upperBound, fileName):
    bufSize = 0;
    f = open(fileName,'ab',bufSize);
#    for reducedFeature in range(10,upperBound+1,(upperBound/100)*10):
    for reducedFeature in range(1,upperBound):
        print "**************** "+ str(reducedFeature) +" ********************"; 
        
        projMatrix = gf.genProjMatrix(eigenvectors,reducedFeature);
        #projMatrix = genProjMatrix(noisyEigvectors,reducedFeature);
        #print projMatrix;
        """
        Apply PCA on training data 
        """
        gf.genProjData(trainingFilePath,trainingReducedPath,cMin,cMax,projMatrix);
        
        """
        Apply PCA on testing data
        """
        gf.genProjData(testingFilePath,testingReducedPath,cMin,cMax,projMatrix);
        
        """
        SVM training and testing
        """
        cmd1 = "./csv2libsvm.py "+trainingReducedPath +" "+formattedTraining;
        cmd2 = "./csv2libsvm.py "+testingReducedPath +" "+formattedTesting;
        cmd3 = "./easy.py " + formattedTraining + " " + formattedTesting;
        #os.system("./csv2libsvm.py ./ionosphere_training_reduced_features ./ionosphere_training_reduced_features_formatted");
        #os.system("./csv2libsvm.py ./ionosphere_testing_reduced_features ./ionosphere_testing_reduced_features_formatted");
        #os.system("./easy.py ./ionosphere_training_reduced_features_formatted ./ionosphere_testing_reduced_features_formatted");
        
        os.system(cmd1);
        os.system(cmd2);
        os.system(cmd3);
        
        f1score = gf.calcF1Score(testingReducedPath,formattedTesting.split('/')[2]+".predict");
        
        f.write(str(reducedFeature)+','+str(f1score)+'\n');
        
        print "F1 Score: "+ str(f1score);
    print "=========================================";
    f.write('\n');
    f.close();

#ionosphere,diabetes, australian,german,colon-cancer;
#dataType = str(sys.argv[1]);
dataType = "tic";
filePath = "./input/"+dataType+"_prePCA";
trainingFilePath = filePath+"_training";
testingFilePath = filePath+"_testing";
trainingReducedPath = "./input/"+dataType+"_training_reduced_features";
testingReducedPath = "./input/"+dataType+"_testing_reduced_features";
formattedTraining = trainingReducedPath+"_formatted";
formattedTesting = testingReducedPath+"_formatted";
            
numOfExpr = 10;

for i in range(0,numOfExpr):         
    # 1) Splitting data into training and testing, trainingSamples:testingSamples = 9:1.
    gf.genTrainingTestingData(filePath,trainingFilePath,testingFilePath);
    
    # 2) Generating Eigenvectors on the training samples.
    # 2.1) Generating the normal Eigenvectors(non-noisy).
    # 2.1.1) Scale data, then generating the covariance matrix.
    cMin,cMax,covMatrix = gf.genCovMatrix(trainingFilePath);
    print "**********";
    start = timeit.default_timer();
    # 2.1.2) Generating Eigenvectors from covariance matrix, the corresponding eigenvalues are in descending order.
    sortedW,eigenvectors = gf.genEigenvectors(covMatrix);
    print "EVD takes "+ str(timeit.default_timer()-start);
    #print sortedW;
    #print eigenvectors[:,0];
    
    # 2.1.3) Getting projection matrix from Eigenvectors based on the different required energy.
    energy = 0.9;
    upperBound = gf.getNumberOfPrinciples(sortedW,energy);    
    print "number of component: " + str(upperBound);
    
    # 2.1.4) Applying projection matrix on training and testing data, then do SVM classification.
    pca_svm_classification(cMin, cMax, eigenvectors, upperBound,'f1score_'+dataType);

    # 2.2) Generating the noisy Eigenvectors, the noise is added into the original covariance matrix.
    # 2.2.2) Generating the noisy Eigenvectors directly. 
    sortedW,noisyEigvectors = gf.genNoisyEigenvectors(covMatrix);
    
    # 2.2.3) Generating the projection matrix based on energy.
    upperBound = gf.getNumberOfPrinciples(sortedW,energy);
    print "number of noisy component: " + str(upperBound);
    pca_svm_classification(cMin, cMax, noisyEigvectors, upperBound,'f1score_'+dataType+'_noisy');
