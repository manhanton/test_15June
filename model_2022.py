# import library

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 


# Initializing the classifiers 

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()
ada = AdaBoostClassifier()

classifiers = [lda,qda,gnb,mnb,bnb,knn,rf,ada]
classifier_names = ["LDA","Quadratic","GaussianNB","MultinomialNB","BernoulliNB","KNeighbors","Random Forests","Ada Boost"]

index = np.arange(len(classifier_names))  

# #Defines the Recursive Feature Selector for best feature selection 

def recursiveFeatureSelector(classifier_model,train_data,train_labels,test_data,number_of_features):
    
    rfe = RFE(classifier_model,number_of_features)
    transformed_train_data = rfe.fit_transform(train_data,train_labels)
    transformed_test_data = rfe.transform(test_data)
    
    return transformed_train_data,transformed_test_data 

#Defines the recursive feature selector for choosing the best feature using Cross Validation   
    
def recursiveFeatureSelectorCV(classifier_model,train_data,train_labels,test_data,number_of_features):
    
    rfe = RFECV(classifier_model,number_of_features)
    transformed_train_data = rfe.fit_transform(train_data,train_labels)
    transformed_test_data = rfe.transform(test_data)
    
    return transformed_train_data,transformed_test_data

#Iterating over all feature preprocessors and classifiers in turn 

performance = list([])

for classifier,classifier_name in zip(classifiers,classifier_names):
    print(classifier)
    X_train=df.iloc[:,0:671]
    X_test=df.iloc[:,0:671]
    y_train=df[['1m_close_future_pct']].astype(int)
    y_test=df[['1m_close_future_pct']].astype(int)

    train_data,test_data = recursiveFeatureSelector(classifier,X_train,y_train,X_test,8)
    classifier.fit(X_train,y_train)
    predicted_labels = classifier.predict(X_test)
    performance.append(metrics.accuracy_score(y_test,predicted_labels))

    print("\n -----------------------------------------------------------------")
    print("Accuracy for ",classifier_name," : ",metrics.accuracy_score(y_test,predicted_labels))
    print("Confusion Matrix for ",classifier_name, " :\n ",metrics.confusion_matrix(y_test,predicted_labels))

    #Cross Validation Scores for each classifier

    scores = cross_val_score(classifier,X_train,y_train,cv=5,scoring='f1_weighted')
    print("Score Array for ",classifier_name, " : \n",scores)
    print("Mean Score for ", classifier_name, " : ", scores.mean())
    print("Stanard Deviation of Scores for ", classifier_name, " : ", scores.std())
    print("-------------------------------------------------------------------\n")