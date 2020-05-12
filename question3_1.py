
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from collections import Counter


# In[2]:


# np.shape(spectrain_x)


# In[3]:


spectrain_x = pickle.load(open("../spec_noise/specfeatures1.pickle",'rb'))
print(np.shape(spectrain_x))
spectrain_x = np.reshape(spectrain_x, (np.shape(spectrain_x)[0], np.shape(spectrain_x)[1]*np.shape(spectrain_x)[2] ))
spectrain_y = pickle.load(open("../spec_noise/specfeatureslabels1.pickle",'rb'))
# spectrain_y = label_binarize(spectrain_y, classes=np.arange(0,10))


# In[4]:


# classifier = OneVsRestClassifier(svm.LinearSVC())#Linear
classifier = pickle.load(open("../spec_noise/spectrogrammodel.pickle","rb"))


# In[5]:


# classifier.fit(spectrain_x, spectrain_y)


# In[6]:


specval_x = pickle.load(open("../spec_noise/specfeaturesvalidation.pickle",'rb'))
specval_x = np.reshape(specval_x, (np.shape(specval_x)[0], np.shape(specval_x)[1]*np.shape(specval_x)[2] ))
specval_y = pickle.load(open("../spec_noise/specfeatureslabelsvalidation.pickle",'rb'))
# specval_y = label_binarize(specval_y, classes=np.arange(0,10))


# In[7]:


specval_pred = classifier.predict(specval_x)
spectrain_pred = classifier.predict(spectrain_x)
# specval_pred = np.argmax(specval_pred,axis=1)
# specval_y = np.argmax(specval_y, axis = 1)
# spectrain_pred = np.argmax(spectrain_pred,axis=1)
# spectrain_y = np.argmax(spectrain_y, axis = 1)


# In[8]:


def confusion_matrixb(ytrue, ypredict, classes):
    instcount = np.shape(ytrue)[0]
    mtx = np.zeros((classes, classes)).astype(int)
    for i in range(instcount):
        if(ytrue[i]==ypredict[i]):
            mtx[ytrue[i],ytrue[i]]+=1
        else:
            mtx[ytrue[i],ypredict[i]]+=1
    t=[0,1,2,3,4,5,6,7,8,9]
    s=np.array(mtx)
    plt.matshow(s, cmap=plt.cm.gray_r)
    plt.colorbar()
    tick_marks = np.arange(classes)
    plt.xticks(tick_marks, t)
    plt.yticks(tick_marks, t)
    import itertools
    thresh = mtx.max() / 2
    for i, j in itertools.product(range(mtx.shape[0]), range(mtx.shape[1])):
        plt.text(j, i, "{:,}".format(mtx[i, j]),horizontalalignment="center",color="white" if mtx[i, j] > thresh else "black")
    #plt.savefig('C:/Users/SajagAgarwal/Desktop/WinSem2019/SML/A2/images/Correlated/ConfusionMatrix/Train/risk1.png')
    plt.show()
    return mtx


# In[9]:


confusion_matrix1 = confusion_matrixb(specval_y,specval_pred, 10)
confusion_matrix2 = confusion_matrixb(spectrain_y,spectrain_pred, 10)


# In[10]:


def precision_(arr):
    lis = []
    for i in range(len(arr)):
        lis.append(arr[i][i] / np.sum(arr[:,i]))
    lis = [round(x,2) for x in lis]
    return lis
def recall_(arr):
    lis = []
    for i in range(len(arr)):
        lis.append(arr[i][i] / np.sum(arr[i,:]))
    lis = [round(x,2) for x in lis]
    return lis
def f1score(precision, recall):
    lis = []
    for i in range(len(precision)):
        lis.append((2*precision[i]*recall[i] )/ (precision[i] + recall[i]))
    lis = [round(x,2) for x in lis]
    return lis


# In[11]:


print(confusion_matrix1)
print(confusion_matrix2)


# In[12]:


print(precision_(confusion_matrix1))
print(recall_(confusion_matrix1))
print(f1score(precision_(confusion_matrix1),recall_(confusion_matrix1)))


# In[13]:


print(precision_(confusion_matrix2))
print(recall_(confusion_matrix2))
print(f1score(precision_(confusion_matrix2),recall_(confusion_matrix2)))


# In[14]:


def average_precision(ytrue, precision):
    a = Counter(ytrue)
    b = len(ytrue)
    count = 0
    for i in range(len(precision)):
        count += precision[i]*a[i]/b
    return round(count,2)


# In[15]:


# print(np.sum(precision_(confusion_matrix1))/10)
# print(np.sum(recall_(confusion_matrix1))/10)
# print(np.sum(f1score(precision_(confusion_matrix1),recall_(confusion_matrix1)))/10)

print(average_precision(specval_y, precision_(confusion_matrix1)))
print(average_precision(specval_y, recall_(confusion_matrix1)))
print(average_precision(specval_y, f1score(precision_(confusion_matrix1),recall_(confusion_matrix1))))


# In[16]:


print(average_precision(spectrain_y, precision_(confusion_matrix2)))
print(average_precision(spectrain_y, recall_(confusion_matrix2)))
print(average_precision(spectrain_y, f1score(precision_(confusion_matrix2),recall_(confusion_matrix2))))


# In[17]:


def accuracy(confusion_matrix):
    count = 0
    for i in range(len(confusion_matrix)):
        count += confusion_matrix[i][i]
    return round(count/np.sum(confusion_matrix),2)
print(accuracy(confusion_matrix1))
print(accuracy(confusion_matrix2))


# In[18]:


# pickle.dump(classifier, open("spectrogrammodel.pickle","wb"))

