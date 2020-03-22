# imports:

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# functions:

def wave_to_stft(
    wave, 
    f_s, win_length, 
    hop_length, n_fft):
    
    # short-time fourier transform:
    
    stft = librosa.stft(wave, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    stft_magnitude, stft_phase = librosa.magphase(stft)
    stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    
    # create axes:
    
    n_f = stft.shape[0]
    n_t = stft.shape[1]

    max_t = (len(wave)-1)/f_s
    f_stft = np.linspace(0, f_s/2, n_f)
    t_stft = np.linspace(0, max_t, n_t)
    
    # organize:
    stft = {
        's': stft_magnitude_db,
        'f': f_stft,
        't': t_stft
    }
    
    return stft


def wave_to_mel(wave, f_s, n_fft, hop_length,
                fmin, fmax, n_mels):
    
    # transform - mel spectrogram:

    mel_spec = librosa.feature.melspectrogram(
        wave, 
        n_fft=n_fft, hop_length=hop_length, 
        n_mels=n_mels, sr=f_s, fmin=fmin, fmax=fmax, 
        power=1)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    
    # create axes:
    
    max_t = (len(wave)-1)/f_s
    n_t = mel_spec.shape[1]
    f_mel = librosa.core.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False)
    t_mel = np.linspace(0, max_t, n_t)
    
    # organize:
    
    mel = {
        's': mel_spec_db,
        'f': f_mel,
        't': t_mel
    }
    
    return mel




