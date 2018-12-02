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
from librosa.effects import trim

# Define notes on Piano
A0 = 27.5
C2 = 65.41
C7 = 2093.00
C8 = 4186.01
# Loads data from .wav file specified
def Load():
    filename = askopenfilename()
    data, sr = sf.read(filename)
    return data, sr


# Given the .wav data, cuts stereo if needed and trims silence
def Prepare(data):
    T = data.T
    if np.shape(T)[0] == 2:
        data = T[0]
    data = trim(data, 40)[0]
    return data


# Once data is Prepared, input it with the sr to return Mel_Transformed spectrogram
# If feeding into a NN for a single chord take the average frame of this
def Mel_transform(data, sr):
    Mel = melspectrogram(
        data, sr, fmin=C2, fmax=C7, n_mels=100, n_fft=2048 * 2, hop_length=1024
    )
    N_Mel = normalize(Mel, norm="l1", axis=0)
    return pow(N_Mel, .25)


# Grabs data, prepares, and transforms in one function
def Get_Mel():
    data, sr = Load()
    data = Prepare(data)
    Mel = Mel_transform(data, sr)
    return Mel


# Given number of classes and a classified sequence from GMM,
# Creates an image to plot
def Plot_Prog(k, Prog):
    N = len(Prog)
    S = np.zeros((k, N))
    for i in range(N):
        S[Prog[i]][i] = 1
    plt.imshow(S, aspect=10)
    plt.show()
