# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 12:53:05 2019

@author: Jayneel
"""

#import pandas and numpy
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

%pwd
df = pd.read_csv('caravan_insurance(1).csv')

df.info()
df[df.columns[0:8]].describe()
df1=df[df.columns[0:9]]
df1

#split data into test and train
from sklearn.model_selection import train_test_split

y=df["caravan"] #Target variable
x=df.drop("caravan",axis=1) #independent variable

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=123)

#logistic regression
import statsmodels.api as sm
logit_model=sm.Logit(yTrain,xTrain)

result=logit_model.fit()

print(result.summary())

#logistic reg: roc,auc & cofusion matrix
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(xTrain,yTrain)

#prediction
lg_pred=logmodel.predict(xTest)

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(yTest,lg_pred)

cmlog=confusion_matrix(yTest,lg_pred)

cmlog[0,1]
np.sum(cmlog)
acculog=(cmlog[0,0]+cmlog[1,1])/np.sum(cmlog)
acculog

from sklearn.metrics import roc_auc_score
from sklearn.metrics import  roc_curve

lg_roc_auc=roc_auc_score(yTest,logmodel.predict(xTest))
lg_roc_auc

###########################
#new model
y=df["caravan"]
z=df[["Contribution car policies","Contribution life insurances","Contribution fire policies","Number of boat policies"]]
zTrain,zTest,yTrain,yTest=train_test_split(z,y,test_size=0.2,random_state=123)

#logistic regression
import statsmodels.api as sm
new_logit_model=sm.Logit(yTrain,zTrain)

new_result=new_logit_model.fit()

print(new_result.summary())

#logistic reg: roc,auc & cofusion matrix
from sklearn.linear_model import LogisticRegression

new_logmodel=LogisticRegression()
new_logmodel.fit(zTrain,yTrain)

#prediction
lg_pred2=new_logmodel.predict(zTest)

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(yTest,lg_pred2)

cmlog2=confusion_matrix(yTest,lg_pred2)

cmlog[0,1]
np.sum(cmlog)
acculog=(cmlog2[0,0]+cmlog2[1,1])/np.sum(cmlog2)
acculog

from sklearn.metrics import roc_auc_score
from sklearn.metrics import  roc_curve

lg_roc_auc2=roc_auc_score(yTest,new_logmodel.predict(zTest))
lg_roc_auc2


###########################
#random forest model
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100,oob_score=True,random_state=123)
random_forest.fit(xTrain,yTrain)
Y_predication = random_forest.predict(xTest)
random_forest.score(xTrain,yTrain)
cmlog2=confusion_matrix(yTest,Y_predication)
print(cmlog2)

acculog2=(cmlog2[0,0]+cmlog2[1,1])/np.sum(cmlog2)
acculog2

lg_roc_auc2=roc_auc_score(yTest,random_forest.predict(xTest))
lg_roc_auc2

random_forest.predict(xTest)
random_forest.predict_proba(xTest)
fpr1,tpr1,thresholds1= roc_curve(yTest,random_forest.predict_proba(xTest)[:,1])
plt.figure()
plt.plot(fpr1,tpr1,label='Logistic regression(area = %0.2f)'% lg_roc_auc2)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive rate')
plt.ylabel('true positive rate')
plt.title('roc & auc')
plt.legend(loc="lower right")
plt.show()


############################################
#SUPPORT VECTOR MACHINE (SVM)
from sklearn import svm,datasets
from sklearn.model_selection import GridSearchCV

#tuning parameters
Cs=[0.001,0.01,0.1,1,10]
gammas=[0.001,0.01,0.1,1]
param_grid ={'C':Cs,'gamma':gammas}
param_grid

grid_search = GridSearchCV(svm.SVC(kernel='rbf'),param_grid,cv=5)
grid_search.fit(xTrain,yTrain)
print(grid_search.best_params_)

#fitting svm with best parameter
from sklearn.svm import SVC, LinearSVC
svc=SVC(C=0.001,gamma=0.001,probability=True)
svc.fit(xTrain,yTrain)
Y_pred_SVM=svc.predict(xTest)

from sklearn.metrics import confusion_matrix
cmlog3=confusion_matrix(yTest,Y_pred_SVM)
acculog3=(cmlog3[0,0]+cmlog3[1,1])/np.sum(cmlog3)
acculog3

#ROC and AUC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import  roc_curve

lg_roc_aucSVC=roc_auc_score(yTest,svc.predict(xTest))
lg_roc_aucSVC

fprSVC,tprSVC,thresholdsSVC= roc_curve(yTest,svc.predict_proba(xTest)[:,1])
fprSVC

#plot roc & auc
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fprSVC,tprSVC,label='Logistic regression(area = %0.2f)'% lg_roc_aucSVC)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive rate')
plt.ylabel('true positive rate')
plt.title('roc & auc')
plt.legend(loc="lower right")
plt.show()








