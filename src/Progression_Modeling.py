import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from librosa.feature import mfcc
from hmmlearn.hmm import GaussianHMM as HMM
from hmmlearn.hmm import GMMHMM
from sklearn.mixture import GaussianMixture as GMM
from tkinter.filedialog import askopenfilename
from sklearn.preprocessing import normalize
from librosa.feature import melspectrogram
from librosa import cqt
from librosa import pseudo_cqt
from librosa.effects import trim

#Define notes on Piano
A0 = 27.5
A1 = 55.00
C1 = 32.70
A2 = 110.00
C2 = 65.41
E6 = 1318.51
C7 = 2093.00
C8 = 4186.01

#Loads data from .wav file specified
def Load():
    filename=askopenfilename()
    data, sr = sf.read(filename)

    return data, sr

def Prepare(data):
    T = data.T
    if np.shape(T)[0]==2:
        data = T[0]

    data = trim(data, 30)[0]

    return data


def Mel_transform(data, sr):
    Mel = melspectrogram(data, sr, fmin=10, fmax=E6, n_mels=100, n_fft=2048*3, hop_length=512*3)
    N_Mel = normalize(Mel, norm='l1', axis=0)
    return pow(N_Mel, .25)

def CQT_transform(data, sr):
    CQT = CQT = abs(pseudo_cqt(data, sr, fmin=A1, n_bins=120, bins_per_octave=24, sparsity=.95, window=('kaiser', 10)))
    N_CQT = normalize(CQT, norm='l1', axis=0)
    return N_CQT
    

def Get_Mel():
    data, sr = Load()
    data = Prepare(data)
    Mel = Mel_transform(data, sr)
    return Mel

def Get_CQT():
    data, sr = Load()
    data = Prepare(data)
    CQT = CQT_transform(data, sr)
    return CQT

def Get_Boundaries(Progression):
    N = len(Progression)
    Boundaries = list()
    i = 0
    j = 0
    state = Progression[i]
    while j<N-1:
        j += 1
        next_state = Progression[j]
        if next_state != state or j==N-1:
            if j-i >= 10:
                Boundaries.append([i, j])
                
            i = j
            state = Progression[i]
            
    return Boundaries
            
            
        

def Deaden(M, tau):
    R, C = np.shape(M)
    M_copy = np.copy(M)
    for i in range(R):
        for j in range(C):
            if M_copy[i][j] < tau:
                M_copy[i][j] = 0
    return M_copy


#Given number of classes and a classified sequence,
#Creates an image to plot
def Plot_Prog(k, Prog):
    N = len(Prog)
    S = np.zeros((k, N))
    for i in range(N):
        S[Prog[i]][i] = 1

    plt.imshow(S, aspect=10)
    plt.show()

def Project_Prog(k, Prog):
    N = len(Prog)
    S = np.zeros((k, N))
    for i in range(N):
        S[Prog[i]][i] = 1

    return S
