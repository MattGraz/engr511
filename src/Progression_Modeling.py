import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from librosa.feature import mfcc
from hmmlearn.hmm import GaussianHMM as HMM
from hmmlearn.hmm import GMMHMM
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import NMF
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
    CQT  = abs(pseudo_cqt(data, sr, fmin=A1, n_bins=60, bins_per_octave=12, sparsity=.95, window=('kaiser', 10)))
    N_CQT = normalize(CQT, norm='l1', axis=0)
    return N_CQT

def Chroma_transform(data, sr):
    CQT = CQT_transform(data, sr)
    Chroma_CQT = Chroma(CQT)
    N_Chroma = normalize(Chroma_CQT, norm='l1', axis=0)
    return Clean(N_Chroma, .1)

def Clean(M, tau):
    Shape = np.shape(M)
    M_Copy = np.copy(M)
    Zeros = np.zeros(Shape[1])

    for i in range(Shape[0]):
        if np.max(M[i]) < tau:
            M_Copy[i] = Zeros

    return normalize(M_Copy, norm='l1', axis=0)


def V_Chroma(CQT_Vector):
    Chroma = np.zeros(12)
    [A, B_b, B, C, C_s, D, E_b, E, F, F_s, G, A_b] \
    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

   
    for i in range(5):
        Chroma[0]  += CQT_Vector[C   + 12*i]
        Chroma[1]  += CQT_Vector[C_s + 12*i]
        Chroma[2]  += CQT_Vector[D   + 12*i]
        Chroma[3]  += CQT_Vector[E_b + 12*i]
        Chroma[4]  += CQT_Vector[E   + 12*i]
        Chroma[5]  += CQT_Vector[F   + 12*i]
        Chroma[6]  += CQT_Vector[F_s + 12*i]
        Chroma[7]  += CQT_Vector[G   + 12*i]
        Chroma[8]  += CQT_Vector[A_b + 12*i]
        Chroma[9]  += CQT_Vector[A   + 12*i]
        Chroma[10] += CQT_Vector[B_b + 12*i]
        Chroma[11] += CQT_Vector[B   + 12*i]

    return Chroma

def Chroma(CQT):
    N = np.shape(CQT)[1]

    Chroma_CQT = np.zeros((12, N))

    for i in range(N):
        Chroma = V_Chroma(CQT[:, i])
        Chroma_CQT[:, i] = Chroma

    return Chroma_CQT
        
def Progression(X):
    S = list()
    H = list()
    for i in range(2,10):
        hmm = HMM(i, random_state=1)
        try:
            hmm.fit(X.T)
            p = hmm.score(X.T)
            H.append(hmm)
        except:
            p = float('-inf')
            H.append(None)
            
        S.append([i, p])
        print(i)
        print(p)
        print()
       

    K = max(S, key = lambda x: x[1])[0]
    return [K, H[K-2].predict(X.T)]

def HMM_Stack_Transform(k, X):
    hmm = HMM(k, random_state=1)

    hmm.fit(X.T)
    P = hmm.predict(X.T)

    C = Compress_Progression(P)

    ST = np.zeros((np.shape(hmm.means_)[1], len(C)))
    for i in range(len(C)):
        ST[:, i] = hmm.means_[C[i]]

    return normalize(ST,'l1', 0)

    

def NMF_Transform(k, X):
    nmf = NMF(k)
    nmf.fit(X)
    T = nmf.components_
    
    return Construct_Progression(T)

def NMF_Stack_Transform(k, X):
    nmf = NMF(k)
    nmf.fit(X)
    V = nmf.transform(X)
    T = nmf.components_
    P = Construct_Progression(T)
    F = Flatten_Progression(P)
    C = Compress_Progression(F)

    ST = np.zeros((np.shape(V)[0], len(C)))

    for i in range(len(C)):
        ST[:, i] = V[:, C[i]]
        
    return normalize(ST,'l1', 0)
    


def Construct_Progression(T):
    Shape = np.shape(T)

    Progression = np.zeros(Shape)
    for i in range(Shape[1]):
        MAX = np.max(T[:,i])
        for j in range(Shape[0]):
            if T[j][i] == MAX:
                Progression[j][i]=1

    return Progression
        

def Flatten_Progression(P):
    Shape = np.shape(P)
    F = np.zeros(Shape[1])

    for i in range(Shape[1]):
        for j in range(Shape[0]):
            if P[j][i] == 1:
                F[i] = j
    return F.astype(int)

def Compress_Progression(F):
    N = len(F)
    Stack = list()
    MIN_LENGTH = .1 * N

    C_0 = F[0]
    j = 0
    i = 0
   
    while j < N-1:
        j+=1
        C = F[j]
        if C != C_0 or j == N-1:
            if j-i >= MIN_LENGTH:
                Stack.append(C_0)
            i=j
            C_0=C
            
    return np.array(Stack)


def NMF_HMM(N, X):
    nmf = NMF(N+4)
    hmm = HMM(N)

    nmf.fit(X)
    Transition = nmf.components_

    hmm.fit(Transition.T)
    P = hmm.predict(Transition.T)

    return (P, Transition)
    
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

def Get_Chroma():
    data, sr = Load()
    data = Prepare(data)
    Chroma_CQT = Chroma_transform(data, sr)
    return Chroma_CQT

def Get_Boundaries(Progression, alpha):
    N = len(Progression)
    Cap = alpha*N
    Boundaries = list()
    i = 0
    j = 0
    state = Progression[i]
    while j<N-1:
        j += 1
        next_state = Progression[j]
        if next_state != state or j==N-1:
            if j-i >= Cap:
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
