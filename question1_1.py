
# coding: utf-8

# In[ ]:

import numpy as np
import glob
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
from librosa import display
import os
from word2number.w2n import word_to_num
import pickle
from copy import deepcopy


# In[ ]:


def hanning(length):
    arr = []
    for i in range(length):
        k = 0.5*(1 - np.cos(2*np.pi*i/(length-1)) )
        arr.append(k)
    return np.reshape(np.array(arr),(np.shape(arr)[0],1))
def dft(x1):
    x = np.ndarray.astype(x1,np.float64)
    n = np.array(range(np.shape(x)[0]))
    nT = np.reshape(n, (np.shape(x)[0], 1))
    a = -2j*np.pi*nT*n/np.shape(x)[0]
    f = np.dot(np.exp(a),x)
    f = f[:int(len(f)/2 + 1)]
    return f
def dftnew(x1):
    x = np.ndarray.astype(x1,np.float64)
    n = np.array(range(len(x)))
    nT = np.reshape(n, (len(x), 1))
    a = -2j*np.pi*nT*n/len(x)
    f = np.dot(np.exp(a),x)
    return f
def FFT(x):
    f = FFT_(x)
    f = f[:int(len(f)/2 + 1)]
    return f
def FFT_(x):
    N = x.shape[0]
    if np.shape(x)[0] <= 32:
        return dftnew(x)
    else:
        Xeven = FFT_(x[::2,:])
        Xodd = FFT_(x[1::2,:])
        f = np.exp(-2j*np.pi*np.arange(N)/N)
        k1 = Xeven + np.dot(np.diag(f[:int(N/2)]) ,Xodd)
        k2 = Xeven + np.dot(np.diag(f[int(N/2):]) ,Xodd)
        return np.vstack((k1, k2))

# In[ ]:


specfeatures = []
labels = []
folders = os.listdir("./training")
for fol in folders:
    namefol = "./training/" + fol + "/*.wav"
    noisefiles = glob.glob("./_background_noise_/*.wav")
    audiofiles = glob.glob(namefol)
    audiofiles = sorted(audiofiles)
    for num in range(len(audiofiles)):
        name = audiofiles[num].split('/')[2]
        noise = np.random.randint(0, len(noisefiles), 1)
        for no in noise:
            fs,data = wavfile.read(audiofiles[num])
            data = deepcopy(data)
            fsnoise, datanoise = wavfile.read(noisefiles[no])
            index1 = np.random.randint(0,len(datanoise),size = 100)
            index2 = np.random.randint(0,len(data),size = 100)
            data[index2] = data[index2] + datanoise[index1]
            data = data/np.max(data)
#             data, fs = librosa.load(audiofiles[num], sr=None, mono=True, offset=0, duration=None)
            time_ms = 20
            winlen = int(time_ms*0.001*fs)
            overlap = 0.5
            reqindices = np.arange(0,len(data), int((1-overlap)*winlen))
            req_pad = [0]*(reqindices[len(reqindices)-1] + winlen - len(data))
            data = list(data)
            data.extend(req_pad)
            data = np.array(data)
            arr = []
            for i in reqindices[:len(reqindices)]:
                arr.append(data[i:i+winlen])
            arr = np.asarray(arr)
            arr = arr.T*hanning(winlen)
            fftarr = dft(arr)
            fftarr = np.abs(fftarr)**2
            if(np.shape(fftarr)[1]>100):
                fftarr = fftarr[:,:100]
            elif(np.shape(fftarr)[1]<100):
                fftarr = np.hstack([fftarr, np.zeros((len(fftarr),100-np.shape(fftarr)[1]))])
            specfeatures.append(fftarr)
            labels.append(word_to_num(fol))
pickle.dump(specfeatures, open('specfeatures1.pickle','wb'))
pickle.dump(labels, open('specfeatureslabels1.pickle','wb'))


# In[ ]:


specfeatures = []
labels = []
folders = os.listdir("./validation")
for fol in folders:
    namefol = "./validation/" + fol + "/*.wav"
    audiofiles = glob.glob(namefol)
    audiofiles = sorted(audiofiles)
    for num in range(len(audiofiles)):#len(audiofiles)
        name = audiofiles[num].split('/')[2]
        data, fs = librosa.load(audiofiles[num], sr=None, mono=True, offset=0, duration=None)
#         fs, data = wavfile.read(audiofiles[num])
        time_ms = 20
        winlen = int(time_ms*0.001*fs)
        overlap = 0.5
        reqindices = np.arange(0,len(data), int((1-overlap)*winlen))
        req_pad = [0]*(reqindices[len(reqindices)-1] + winlen - len(data))
        data = list(data)
        data.extend(req_pad)
        data = np.array(data)
        arr = []
        for i in reqindices[:len(reqindices)]:
            arr.append(data[i:i+winlen])
        arr = np.asarray(arr)
        arr = arr.T*hanning(winlen)
        fftarr = dft(arr)
        fftarr = np.abs(fftarr)**2
        if(np.shape(fftarr)[1]>100):
            fftarr = fftarr[:,:100]
        elif(np.shape(fftarr)[1]<100):
            fftarr = np.hstack([fftarr, np.zeros((len(fftarr),100-np.shape(fftarr)[1]))])
#         print(np.shape(fftarr))
        specfeatures.append(fftarr)
        labels.append(word_to_num(fol))
pickle.dump(specfeatures, open('specfeaturesvalidation.pickle','wb'))
pickle.dump(labels, open('specfeatureslabelsvalidation.pickle','wb'))


# In[ ]:





# In[ ]:





# In[1]:


import numpy as np
import glob
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
from librosa import display
# import IPython.display as ipd
import os
from word2number.w2n import word_to_num
import pickle
def hanning(length):
    arr = []
    for i in range(length):
        k = 0.5*(1 - np.cos(2*np.pi*i/(length-1)) )
        arr.append(k)
    return np.reshape(np.array(arr),(np.shape(arr)[0],1))
def dft(x1):
    x = np.ndarray.astype(x1,np.float64)
    n = np.array(range(np.shape(x)[0]))
    nT = np.reshape(n, (np.shape(x)[0], 1))
    a = -2j*np.pi*nT*n/np.shape(x)[0]
    f = np.dot(np.exp(a),x)
    f = f[:int(len(f)/2 + 1)]
    return f

def dftnew(x1):
    x = np.ndarray.astype(x1,np.float64)
    n = np.array(range(len(x)))
    nT = np.reshape(n, (len(x), 1))
    a = -2j*np.pi*nT*n/len(x)
    f = np.dot(np.exp(a),x)
    return f
def FFT(x):
    f = FFT_(x)
    f = f[:int(len(f)/2 + 1)]
    return f
def FFT_(x):
    N = x.shape[0]
    if np.shape(x)[0] <= 32:
        return dftnew(x)
    else:
        Xeven = FFT_(x[::2,:])
        Xodd = FFT_(x[1::2,:])
        f = np.exp(-2j*np.pi*np.arange(N)/N)
        k1 = Xeven + np.dot(np.diag(f[:int(N/2)]) ,Xodd)
        k2 = Xeven + np.dot(np.diag(f[int(N/2):]) ,Xodd)
        return np.vstack((k1, k2))
# In[2]:

#Ref for FFT and DFT: https://towardsdatascience.com/fast-fourier-transform-937926e591cb
from copy import deepcopy
audiofiles = glob.glob("../training/zero/*.wav")
noisefiles = glob.glob("../_background_noise_/*.wav")
noise = np.random.randint(0, len(noisefiles), 2)
num = 0
# for no in noise:
#     fs,data = wavfile.read(audiofiles[num]) 
#     data = deepcopy(data)
#     fsnoise, datanoise = wavfile.read(noisefiles[no])
#     index1 = np.random.randint(0,len(datanoise),size = 100)
#     index2 = np.random.randint(0,len(data),size = 100)
#     data[index2] = data[index2] + datanoise[index1]
#     data = data/np.max(data)
# datanoise = datanoise/np.max(datanoise)
data,fs = librosa.load(audiofiles[num], sr=None, mono=True, offset=0, duration=None)
# ipd.Audio(data, rate=fs)
# plt.figure()
# librosa.display.waveplot(y=data, sr=fs)
# plt.show()


# In[3]:


# winlen = 128
time_ms = 20
winlen = int(time_ms*0.001*fs)
print(winlen)
overlap = 0.5
reqindices = np.arange(0,len(data), int((1-overlap)*winlen))
req_pad = [0]*(reqindices[len(reqindices)-1] + winlen - len(data))
data = list(data)
data.extend(req_pad)
data = np.array(data)
arr = []
for i in reqindices[:len(reqindices)-1]:
    arr.append(data[i:i+winlen])


# In[4]:


arr = np.asarray(arr)
arr = arr.T*hanning(winlen)


# In[7]:


# audiofiles[num]


# In[5]:


# fftarr = np.fft.rfft(arr, axis=0)#https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html - why rfft and not fft
fftarr = dft(arr)
fftarr = np.abs(fftarr)**2
#         plt.imshow(fftarr, aspect="auto")
#         plt.colorbar()
#         plt.imshow()


# In[6]:


plt.imshow(fftarr, aspect="auto")
plt.colorbar()

yticks = np.linspace(0,np.shape(fftarr)[0],10) #ticks
ylabels = yticks*fs/np.shape(arr)[0] #nth bin in FFT = n*sampling frquency/size of fft
ylabels = list(map(int, ylabels)) #labels
plt.yticks(yticks,ylabels)
plt.ylabel("Frequency (Hz)")
plt.gca().invert_yaxis()

# xticks = np.linspace(0, np.shape(fftarr)[1], 10)#ticks
# # newlabels = ["{:0.2f}".format(x*len(fftarr)/fs) for x in xticks]
# labels =["{:0.2f}".format((x*len(data)/np.shape(fftarr)[1])/fs) for x in xticks]#
# plt.xticks(xticks, labels)
plt.xlabel("Time window")
plt.title("Spectrogram")
plt.show()

