# imports:

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# config:

# f_s = 44100 # [Hz]
f_s = 48000 # [Hz]

params = {}

params['stft'] = {
    'f_s': f_s, 
    'n_fft': 4096,
    'win_length': 4096, 
    'hop_length': 512, 
    
}

params['mel'] = {
    'f_s': f_s, 
    'n_fft': 4096,
    'hop_length': 512,
    'fmin': 0,
    'fmax': 10000,
    'n_mels': int(6*64),
}


# functions:

def apply(x_wave, func, func_params):
    
    x_spec = func(x_wave, **func_params)
    
    return x_spec