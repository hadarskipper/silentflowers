# imports:

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# config:

f_s = 44100 # [Hz]

params_stft = {
    'n_fft': 4096,
    'win_length': 4096, 
    'hop_length': 512, 
    'f_s': f_s, 
}


# functions:

def apply(x_wave, func, func_params):
    
    x_spec = func(x_wave, **func_params)
    
    return x_spec