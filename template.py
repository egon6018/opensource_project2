#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Please write the GITHUB URL below!
#https://github.com/egon6018/opensource_project2.git


# In[22]:


import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[27]:


def load_dataset(dataset_path):
    #Load the csv file at the given path into the pandas DataFrame and return the DataFrame
    dataset_df = pd.read_csv(dataset_path)
    return dataset_df
    
def dataset_stat(dataset_df):
    #For the given DataFrame, return the following statistical analysis results in order
        #Number of features
        #Number of data for class 0
        #Number of data for class 1
    n_feats = data_df.shape[1] - 1
    n_class0 = len(data_df.loc[data_df["target"] == 0])
    n_class1 = len(data_df.loc[data_df["target"] == 1])
    return n_feats, n_class0, n_class1
        
def split_dataset(dataset_df,testset_size):
    #Splitting the given DataFrame and return train data, test data,train label, and test label in order
    #You must split the data using hte given test size
    x = dataset_df.drop(columns="target", axis=1)
    y = dataset_df["target"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testset_size)
    return x_train, x_test, y_train, y_test
    
def decision_tree_train_test(x_train,x_test,y_train,y_test):
    #Using the given train dataset, train the decision tree model
        #You can implement with default arguments
    #After the training, evaluate the performances of the model using the given test dataset
    #Return three performance metrics(accuracy,precision,recall)in order
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, dt_cls.predict(x_test))
    precision = precision_score(y_test, dt_cls.predict(x_test), labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    recall = recall_score(y_test, dt_cls.predict(x_test), labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    return accuracy, precision, recall

def random_forest_train_test(x_train,x_test,y_train,y_test):
    #Using the given train dataset, train the random forest model
        #You can implement with default arguments
    #After the training, evaluate the performances of the model using the given test dataset
    #Return three performance metrics(accuracy,precision,recall)in order
    rf_cls = RandomForestClassifier()
    rf_cls.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, rf_cls.predict(x_test))
    precision = precision_score(y_test, rf_cls.predict(x_test), labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    recall = recall_score(y_test, rf_cls.predict(x_test), labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    return accuracy, precision, recall
    
def svm_train_test(x_train,x_test,y_train,y_test):
    #Using the given train dataset, train the pipeline consists of a standard scaler and SVM
        #You can implement with default arguments
    #After the training, evaluate the performances of the model using the given test dataset
    #Return three performance metrics(accuracy,precision,recall)in order
    svm_pipe = make_pipeline(
        StandardScaler(),
        SVC()
    )
    svm_pipe.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, svm_pipe.predict(x_test))
    precision = precision_score(y_test, svm_pipe.predict(x_test), labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    recall = recall_score(y_test, svm_pipe.predict(x_test), labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    return accuracy, precision, recall
    
def print_performances(acc,prec,recall):
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", recall)


# In[28]:


if __name__ == '__main__':
    data_path = sys.argv[1]
    data_df = load_dataset(data_path)
    
    n_feats, n_class0, n_class1 = dataset_stat(data_df)
    print("Number of features ", n_feats)
    print("Number of class 0 data entries: ", n_class0)
    print("Number of class 1 data entries: ", n_class1)
    
    print("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
    x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))
    
    acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
    print("\nDecision Tree Performances")
    print_performances(acc, prec, recall)
    
    acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
    print("\nRandom Forest Performances")
    print_performances(acc, prec, recall)
    
    acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
    print("\nSVM Performances")
    print_performances(acc, prec, recall)

