
# coding: utf-8

# In[1]:

import numpy as np
from sklearn import preprocessing,svm
from sklearn.pipeline import Pipeline
import nltk
import subprocess
import time
import utility
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


# In[2]:

# Load data
print "creating classifier..."
no_samples = 10000
positive = np.loadtxt("vectors/positive.csv",  delimiter=',')
negetive = np.loadtxt("vectors/negative.csv", delimiter=',')
X = np.concatenate((positive, negetive), axis=0)
p = np.ones((no_samples, 1))
n = np.full((no_samples, 1), -1, dtype=np.int64)
Y = np.concatenate((p,n), axis=0)
y = Y.ravel()
#scale
scaler = preprocessing.StandardScaler()
#classifying
svm_module = svm.SVC(probability=True)
classifier = Pipeline(steps= [('scale', scaler),('svm', svm_module)])#, ('svm', svm_module)])
classifier.fit(X, y)
print "Classifer created"


# In[ ]:

acc = cross_val_score(classifier, X, y, cv=10, scoring='accuracy', n_jobs=-1)
pre = cross_val_score(classifier, X, y, cv=10, scoring='precision', n_jobs=-1)
rec = cross_val_score(classifier, X, y, cv=10, scoring='recall', n_jobs=-1)
roc = cross_val_score(classifier, X, y, cv=10, scoring='roc_auc', n_jobs=-1)
print "Acc: %f +- %f"%(np.mean(acc), np.std(acc))
print "Pre: %f +- %f"%(np.mean(pre), np.std(pre))
print "Rec: %f +- %f"%(np.mean(rec), np.std(rec))
print "AUC: %f +- %f"%(np.mean(roc), np.std(roc))


# In[ ]:

crossValidation = StratifiedKFold(n_splits=10)
mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'blue', 'red', 'brown', 'black'])
lw = 2

i = 0


for (train, test), color in zip(crossValidation.split(X, y), colors):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    #plt.plot(fpr, tpr, lw=lw, color=color, label="ROC fold %d (area = %0.2f)"%(i, roc_auc))
    i += 1
    
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')


mean_tpr /= crossValidation.get_n_splits(X, Y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Best estimator for TF-IDF & LR')
plt.legend(loc="lower right")
#plt.show()

#
# Name your results -
#                  ||
#                  V
np.savetxt("state_of_art-scaler.csv", np.asarray([mean_fpr, mean_tpr]), delimiter=",")



# In[ ]:



