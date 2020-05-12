
# coding: utf-8

# In[ ]:


#PART 1


# In[ ]:


import numpy as np
import glob
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
from librosa import display
import os
from scipy.fftpack import dct
import pickle
from word2number.w2n import word_to_num
from copy import deepcopy


# In[ ]:


def hamming(length):
    arr = []
    for i in range(length):
        k = (0.54 - 0.46*np.cos(2*np.pi*i/(length-1)) )
        arr.append(k)
    return np.array(arr)
def HzToMel(Hz):
    return (2595*np.log10(1 + Hz/700))
def MelToHz(m):
    return (700*( 10**(m/2595) - 1))


# In[ ]:


# audiofiles = glob.glob("./training/one/*.wav")
# audiofiles = sorted(audiofiles)
# np.shape(mfccfeatures)


# In[ ]:


#Read the signal
req = 1
mfccfeatures = []
labels = []
folders = os.listdir("./training")
for fol in folders:
    namefol = "./training/" + fol + "/*.wav"
    audiofiles = glob.glob(namefol)
    audiofiles = sorted(audiofiles)
#     noisefiles = glob.glob("./_background_noise_/*.wav")
    for num in range(len(audiofiles)):#len(audiofiles)
#         print("----------------------------------------")
        name = audiofiles[num].split('/')[2]
#         print(name)
        fs, xdata_num = wavfile.read(audiofiles[num])
#         xdata_num, fs = librosa.load(audiofiles[num], sr=None, mono=True, offset=0, duration=None) #fs - sampling freq
        xdata_num = deepcopy(xdata_num)
#         no = np.random.randint(0, len(noisefiles))
#         fsnoise, datanoise = wavfile.read(noisefiles[no])
#         index1 = np.random.randint(0,len(datanoise),size = 100)
#         index2 = np.random.randint(0,len(xdata_num),size = 100)
#         xdata_num[index2] = xdata_num[index2] + datanoise[index1]
        xdata_num = xdata_num/np.max(xdata_num)
        energy = np.sum(xdata_num**2)
#         plt.figure()
#         librosa.display.waveplot(y=xdata_num, sr=fs)
#         plt.show()

        #Step 1 - pre emphasis
        alpha = 0.95 #95% of any one sample is presumed to originate from previous sample.
        newx = [0]
        newx.extend(xdata_num)
#         print(xdata_num)
        xpre_num = xdata_num - alpha*np.array(newx[:-1])
#         print(xpre_num)


        #Step 2 - framing
        time_ms = 20
        winlen = int(time_ms*0.001*fs)
        overlap = 0.5
        reqindices = np.arange(0,len(xpre_num), int((1-overlap)*winlen))
        #if u dont want to pad
        xframes = []
        for i in reqindices[:len(reqindices)-1]:
            xframes.append(xpre_num[i:i+winlen])
#         print(np.shape(xframes))
        #if u want to pad
        req_pad = [0]*(reqindices[len(reqindices)-1] + winlen - len(xpre_num))
        xpre_num = list(xpre_num)
        xpre_num.extend(req_pad)
        xpre_num = np.array(xpre_num)
        xframes = []
        for i in reqindices[:len(reqindices)]:
            xframes.append(xpre_num[i:i+winlen])
#         print(np.shape(xframes))


        #step 3 - Hamming windowing
        xhamm = xframes*hamming(winlen)
#         print(np.shape(xhamm))


        #step 4 - FFT
        NFFT = 2048#1024
        xfft = np.fft.rfft(xframes, NFFT)
        xpower = (np.abs(xfft))**2
#         if(np.shape(xpower)[1]>100):
#             xpower = xpower[:,:100]
#         elif(np.shape(fftarr)[1]<100):
#             xpower = np.hstack([xpower, np.zeros((len(xpower),100-np.shape(xpower)[1]))])
        xpower = (1/NFFT)*xpower
#         print(np.shape(xfft))
#         print(np.shape(xpower))


        #Step 5 - Mel filter bank processing
        low_frq_Hz = 0 
        high_freq_Hz = fs/2
        num_filters = 40
        high_freq_mel = HzToMel(high_freq_Hz)
        filters_mel = np.linspace(0, high_freq_mel, num_filters+2)
        filters_Hz = MelToHz(filters_mel)
        filter_pts = np.floor((NFFT + 1)*filters_Hz / fs).astype(int)
        melfilters = np.zeros((num_filters, int(NFFT/2 + 1)))
        for n in range(1,num_filters+1):
            melfilters[n-1,filter_pts[n-1]:filter_pts[n]] = np.linspace(0, 1,filter_pts[n]-filter_pts[n-1])
            melfilters[n-1,filter_pts[n]:filter_pts[n+1]] = np.linspace(1, 0,filter_pts[n+1]-filter_pts[n])
#         for mel in melfilters:
#             plt.plot(mel)
#         plt.title('Mel Filter Bank')
#         plt.xlabel('Frequency (in Hz)')
#         plt.ylabel('Amplitude')
#         plt.show()
#         x_filtered = np.dot(melfilters, xpower.T)
        x_filtered = np.dot(xpower, melfilters.T)
        x_filtered[x_filtered==0] = np.finfo(float).eps
        energy = np.reshape(np.sum(x_filtered, axis = 1),(len(x_filtered),1))
        x_filtered = 20 * np.log10(x_filtered)
#         print(np.shape(x_filtered))

        #Step 6 - DCT (Mel Frequency Cepstrum Coefficient)
        u1 = 0
        u2 = 40
        mfcc = dct(x_filtered, type=2, axis=1, norm='ortho')[:,u1:u2]#[:,u2:u1]
        if(np.shape(mfcc)[1]!=u2):
            mfcc = np.hstack([mfcc,np.zeros((num_filters,u1-np.shape(mfcc)[1]))])
        if(np.shape(mfcc)[0]>100):
            mfcc = mfcc[:100]
        elif(np.shape(mfcc)[0]<100):
            mfcc = np.vstack([mfcc, np.zeros((100-len(mfcc),np.shape(mfcc)[1]))])
        mfccfeatures.append(mfcc)
        labels.append(word_to_num(fol))
#         plt.show()
pickle.dump(mfccfeatures, open('1mfccfeatures.pickle','wb'))
pickle.dump(labels, open('1mfccfeatureslabels.pickle','wb'))


# In[ ]:


#Read the signal
req = 1
mfccfeatures = []
labels = []
folders = os.listdir("./validation")
for fol in folders:
    namefol = "./validation/" + fol + "/*.wav"
    audiofiles = glob.glob(namefol)
    audiofiles = sorted(audiofiles)
    for num in range(len(audiofiles)):
        name = audiofiles[num].split('/')[2]
        xdata_num, fs = librosa.load(audiofiles[num], sr=None, mono=True, offset=0, duration=None) #fs - sampling freq
        energy = np.sum(xdata_num**2)

        #Step 1 - pre emphasis
        alpha = 0.95
        newx = [0]
        newx.extend(xdata_num)
        xpre_num = xdata_num - alpha*np.array(newx[:-1])


        #Step 2 - framing
        time_ms = 20
        winlen = int(time_ms*0.001*fs)
        overlap = 0.5
        reqindices = np.arange(0,len(xpre_num), int((1-overlap)*winlen))
        req_pad = [0]*(reqindices[len(reqindices)-1] + winlen - len(xpre_num))
        xpre_num = list(xpre_num)
        xpre_num.extend(req_pad)
        xpre_num = np.array(xpre_num)
        xframes = []
        for i in reqindices[:len(reqindices)]:
            xframes.append(xpre_num[i:i+winlen])

        #step 3 - Hamming windowing
        xhamm = xframes*hamming(winlen)

        #step 4 - FFT
        NFFT = 2048#1024
        xfft = np.fft.rfft(xframes, NFFT)
        xpower = (np.abs(xfft))**2
        xpower = (1/NFFT)*xpower


        #Step 5 - Mel filter bank processing
        low_frq_Hz = 0 
        high_freq_Hz = fs/2
        num_filters = 40
        high_freq_mel = HzToMel(high_freq_Hz)
        filters_mel = np.linspace(0, high_freq_mel, num_filters+2)
        filters_Hz = MelToHz(filters_mel)
        filter_pts = np.floor((NFFT + 1)*filters_Hz / fs).astype(int)
        melfilters = np.zeros((num_filters, int(NFFT/2 + 1)))
        for n in range(1,num_filters+1):
            melfilters[n-1,filter_pts[n-1]:filter_pts[n]] = np.linspace(0, 1,filter_pts[n]-filter_pts[n-1])
            melfilters[n-1,filter_pts[n]:filter_pts[n+1]] = np.linspace(1, 0,filter_pts[n+1]-filter_pts[n])
        x_filtered = np.dot(xpower, melfilters.T)
        x_filtered[x_filtered==0] = np.finfo(float).eps
        energy = np.reshape(np.sum(x_filtered, axis = 1),(len(x_filtered),1))
        x_filtered = 20 * np.log10(x_filtered)

        #Step 6 - DCT (Mel Frequency Cepstrum Coefficient)
        u1 = 0
        u2 = 40
        mfcc = dct(x_filtered, type=2, axis=1, norm='ortho')[:,u1:u2]
        if(np.shape(mfcc)[1]!=u2):
            mfcc = np.hstack([mfcc,np.zeros((num_filters,u1-np.shape(mfcc)[1]))])
        if(np.shape(mfcc)[0]>100):
            mfcc = mfcc[:100]
        elif(np.shape(mfcc)[0]<100):
            mfcc = np.vstack([mfcc, np.zeros((100-len(mfcc),np.shape(mfcc)[1]))])
        mfccfeatures.append(mfcc)
        labels.append(word_to_num(fol))
#         plt.show()
pickle.dump(mfccfeatures, open('1mfccfeaturesvalidation.pickle','wb'))
pickle.dump(labels, open('1mfccfeaturesvalidationlabels.pickle','wb'))
# print(np.shape(mfccfeatures))


# In[ ]:


# np.shape(mfccfeatures)


# In[ ]:


#PART 2


# In[1]:


import numpy as np
import pickle
from sklearn import svm
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt


# In[2]:


mfcctrain_x = pickle.load(open("1mfccfeatures.pickle",'rb'))
mfcctrain_x = np.reshape(mfcctrain_x, (np.shape(mfcctrain_x)[0], np.shape(mfcctrain_x)[1]*np.shape(mfcctrain_x)[2] ))
mfcctrain_y = pickle.load(open("1mfccfeatureslabels.pickle",'rb'))
# mfcctrain_y = label_binarize(mfcctrain_y, classes=np.arange(0,10))
mfccval_x = pickle.load(open("1mfccfeaturesvalidation.pickle",'rb'))
mfccval_x = np.reshape(mfccval_x, (np.shape(mfccval_x)[0], np.shape(mfccval_x)[1]*np.shape(mfccval_x)[2] ))
mfccval_y = pickle.load(open("1mfccfeaturesvalidationlabels.pickle",'rb'))
# mfccval_y = label_binarize(mfccval_y, classes=np.arange(0,10))


# In[3]:


# classifier = OneVsRestClassifier(svm.LinearSVC())
# classifier.fit(mfcctrain_x, mfcctrain_y)
classifier = pickle.load(open("mfcc_40_2048.pickle","rb"))


# In[ ]:


mfccval_pred = classifier.predict(mfccval_x)
mfcctrain_pred = classifier.predict(mfcctrain_x)



# mfccval_pred = np.argmax(mfccval_pred,axis=1)
# mfccval_y = np.argmax(mfccval_y, axis = 1)
# mfcctrain_pred = np.argmax(mfcctrain_pred,axis=1)
# mfcctrain_y = np.argmax(mfcctrain_y, axis = 1)


# In[ ]:


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


# In[ ]:


confusion_matrix1 = confusion_matrixb(mfccval_y,mfccval_pred, 10)


# In[ ]:


# confusion_matrix1


# In[ ]:


confusion_matrix2 = confusion_matrixb(mfcctrain_y,mfcctrain_pred, 10)


# In[ ]:


# confusion_matrix2


# In[ ]:


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


# In[ ]:


print(precision_(confusion_matrix1))
print(recall_(confusion_matrix1))
print(f1score(precision_(confusion_matrix1),recall_(confusion_matrix1)))


# In[ ]:


print(precision_(confusion_matrix2))
print(recall_(confusion_matrix2))
print(f1score(precision_(confusion_matrix2),recall_(confusion_matrix2)))


# In[ ]:


def average_precision(ytrue, precision):
    a = Counter(ytrue)
    b = len(ytrue)
    count = 0
    for i in range(len(precision)):
        count += precision[i]*a[i]/b
    return round(count,2)


# In[ ]:


print(average_precision(mfccval_y, precision_(confusion_matrix1)))
print(average_precision(mfccval_y, recall_(confusion_matrix1)))
print(average_precision(mfccval_y, f1score(precision_(confusion_matrix1),recall_(confusion_matrix1))))


# In[ ]:


print(average_precision(mfcctrain_y, precision_(confusion_matrix2)))
print(average_precision(mfcctrain_y, recall_(confusion_matrix2)))
print(average_precision(mfcctrain_y, f1score(precision_(confusion_matrix2),recall_(confusion_matrix2))))


# In[ ]:


def accuracy(confusion_matrix):
    count = 0
    for i in range(len(confusion_matrix)):
        count += confusion_matrix[i][i]
    return round(count/np.sum(confusion_matrix),2)
print(accuracy(confusion_matrix1))
print(accuracy(confusion_matrix2))


# In[ ]:


import pickle
pickle.dump(classifier,open("mfcc_40_2048.pickle",'wb'))

