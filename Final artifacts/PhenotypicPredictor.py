
# coding: utf-8

# In[1]:


import _pickle as cPickle
import pickle
import pandas as pd
import os
import random
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn import cluster as Kcluster, metrics as Met ,preprocessing as preprocessing
import numpy as np
import pylab as pl
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import argparse
# from tabulate import tabulate


# ## Read the data, flatten and shuffle the data and labels

# In[2]:


def load_data(data_path,label_path):
    global sample_set
    global sample_data
    # Get the sample identifier(accession). Optionally equally subsample from each class label
    sample_set = pd.read_csv(label_path, sep=',', error_bad_lines=False)
    sample_set['combined']=sample_set['population']+sample_set['sequencing_center'].astype(str)
    #Equally Subsample from each of the classses
    # for lab in labels:
    #     SAMPLES_FILETERED = SAMPLES[SAMPLES.label == lab].sample(n=50)
    #     sample_set =sample_set.append(SAMPLES_FILETERED,ignore_index=False)
    sample_set_id = sample_set['accession']
    sample_data = pd.DataFrame()
    root = data_path
    for sampleid in sample_set_id:
        try:
            #ambig_info_df = pd.read_csv(root+"\\"+sampleid+ '\\bias\\aux_info\\ambig_info.tsv',sep='\t',error_bad_lines=False)
            quantified_data = pd.read_csv(root+"//"+sampleid+"//bias//quant2.sf",sep='\t',error_bad_lines=False)
            #quantified_data = pd.DataFrame(pd.concat([quantified_data, ambig_info_df], axis=1).set_index('Name').stack()).transpose()
            quantified_data = pd.DataFrame(quantified_data.set_index('Name').stack()).transpose()
            sample_data = sample_data.append(quantified_data)
        except KeyboardInterrupt:
            break 
    #     sample_data.drop('Length', axis=1, level=1, inplace=True)
    #     pickle.dump(sample_data, open("data_ambig_300.pkl", 'wb'))
    #Shuffle the data and the labels
    shufflelist = list(range(sample_data.shape[0]))
    random.shuffle(shufflelist)
    sample_data=sample_data.iloc[shufflelist,:]
    sample_set=sample_set.iloc[shufflelist,:]


# In[3]:


class predictorROC:   
    #Yreal is the real outcome value, Yscore is the caluclated value
    #threshold is the threshold value to calculate the specificity and sensitivity  
    def draw_roc(self, Yreal, Yscore):
        fpr = []
        tpr = []
        roc_auc = []
        fpr, tpr, _ = roc_curve(Yreal, Yscore)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


# # Dimensionality Reduction Using PCA

# In[4]:


def get_eigenvalues(data):
    U, V, W = np.linalg.svd(data, full_matrices=False)
    return U,V

def perf_pca(data):
    #U, V, W = np.linalg.svd(data, full_matrices=False)
    #newV = [x for x in V if x >= 1]
    # find the number of principal components.
    # select the components with eigenvalue greater than 1.
    #dim = len(newV)
#     sk_pca = sklearnPCA(n_components=mle)
    sk_pca = sklearnPCA()
    pca = sk_pca.fit_transform(data)
    return pca


# In[5]:


def calc_pca():
    #Unable to do normalization for large inputs due to memory issues.
    # Normalize the data
    # sample_norm_data = preprocessing.scale(np.array(sample_data))
    # Compute the correlation matrix
    # sample_corr_data = np.corrcoef(sample_norm_data, rowvar=False)
    #pca_sample = perf_pca(sample_corr_data)
    pca=perf_pca(sample_data)
    # pca = pickle.load(open("pca_ambig_250.pkl",'rb'))
    pca.shape


# ## Preprocessing and Classifier functions

# In[6]:


def perf_logistic_regression(xtr, ytr,xtest):
    logm = LogisticRegression()
    logm.fit(xtr,ytr)
    return logm.predict(xtest)


# In[7]:


def perf_random_forest(xtr,ytr,xtest):
    rf = RandomForestClassifier(n_estimators=250)
    rf.fit(xtr, ytr)
    return rf.predict(xtest)


# In[8]:


def perf_neural_network(xtr,ytr,xtest):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    perf_preprocessing(xtr,xtest)
    clf.fit(xtr,ytr)
    return clf.predict(xtest)


# In[9]:


def perf_preprocessing(xtr,xtest):
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(xtr)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    # Now apply the transformations to the data:
    xtr = scaler.transform(xtr)
    xtest = scaler.transform(xtest)
    return xtr,xtest


# ## Feature Selection using ExtraTreeClassifier

# In[10]:


def perf_ftr_selection(data,label,plotDistr):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    forest.fit(data, label)
    index = data.shape[1]
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]    
    for f in range(data.shape[1]):
#       print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        if importances[indices[f]] == 0:
            index = f-1
            break
    #Plot the feature importance distribution. This plot how many features are important among all the features        
    if plotDistr:
        pl.figure()
        pl.title("Feature importances")
        for tree in forest.estimators_:
            pl.plot(range(index+1000), tree.feature_importances_[indices[:index+1000]], "r")

        pl.plot(range(index+1000), importances[indices[:index+1000]], "b")
        pl.show()
    
    return indices[:index]


# ## Perform Cross Validation 

# In[11]:


def perf_cross_validation(X,y,nfolds):
    kf = KFold(n_splits = nfolds)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        #Select the best features for the current fold
        ftr_indices = perf_ftr_selection(X_train,y_train,False)
        train_data=X_train.iloc[:,ftr_indices]
        test_data = X_test.iloc[:,ftr_indices]
        
        #Print the classification report for each fold
        print ("Train data shape", train_data.shape , "label shape",y_train.shape, "test shape", test_data.shape)
        predictions = perf_random_forest(train_data,y_train,test_data)
        print (classification_report(y_test,predictions))


# In[12]:


def prnt_cross_val_results():
    label_type = ['population','sequencing_center','combined']
    for label in label_type:
        print ("Performing cross validation for label : " , label)
        perf_cross_validation(sample_data,sample_set[label],3)


# ## Create the model dump for each classifier

# In[13]:


def create_model_dump():
    label_type = ['population','sequencing_center','combined']
    for label in label_type:
        indices = perf_ftr_selection(sample_data,sample_set[label],True)
        reduced_data= sample_data.iloc[:,indices]
        rf = RandomForestClassifier(n_estimators=250)
        rf.fit(reduced_data,sample_set[label])
        pickle.dump((indices,rf), open("model_"+label+".pkl", 'wb'))


# In[14]:


#function to parse the input argument
def arg_parser():
    parser = argparse.ArgumentParser('PhenotypicPredictor.py',add_help = True)
    parser.add_argument("-m" , metavar = "<Path to model dump>")
    parser.add_argument("-t" , metavar = "<Path to test sample>")
    parser.add_argument("-l" , metavar = "<Path to test label file>")
    args = parser.parse_args()
    return args.m ,args.t, args.l


# In[15]:


def main():
    model_path, test_data_path, test_label_path = arg_parser()
#     model_path, test_data_path, test_label_path = "/Users/shilpageorge/Desktop/Fall'17/CB/","/Users/shilpageorge/Desktop/Fall'17/CB/test","/Users/shilpageorge/Desktop/Fall'17/CB/p1_train_pop_lab_test_label.csv"
    population_indices,model_population = pickle.load(open(model_path.rstrip("/")+"/model_population.pkl",'rb'))
    combined_indices,model_combined = pickle.load(open(model_path.rstrip("/")+"/model_combined.pkl",'rb'))
    seq_indices,model_seq = pickle.load(open(model_path.rstrip("/")+"/model_sequencing_center.pkl",'rb'))
    #call load_data. The data will be loaded to the global dataframe sample_data
    #labels will be loaded to the global dataframe sample_set
    load_data(test_data_path.rstrip("/"),test_label_path.rstrip("/"))
    
    #Compute the predictions using the model dump for each classifier
    population_predictions = model_population.predict(sample_data.iloc[:,population_indices])
    seq_predictions = model_seq.predict(sample_data.iloc[:,seq_indices])
    combined_predictions = model_combined.predict(sample_data.iloc[:,combined_indices])
    
    
    #Print the classification report for each classifier
    print ( "F1 Score and Accuracy for Classifier label - Population \n")
    print (classification_report(sample_set['population'],population_predictions))
    print ( "F1 Score and Accuracy for Classifier label - Sequencing Centre \n")
    print (classification_report(sample_set['sequencing_center'],seq_predictions))
    print ( "F1 Score and Accuracy for Classifier label - Joint \n")
    print (classification_report(sample_set['combined'],combined_predictions))



# In[ ]:

if __name__ == '__main__':
    main()



