# imports:

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# functions:

def wave_to_stft(wave, f_s, win_length, hop_length, n_fft):
    
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


def wave_to_mel(wave, 
                n_fft=1024, hop_length=512, sample_rate=16000, 
                fmin=10, fmax=10000, n_mels=64, 
                plot_flag=False):

    mel_spec = librosa.feature.melspectrogram(wave, n_fft=n_fft, hop_length=hop_length,
                                              n_mels=n_mels, sr=sample_rate, power=1.0, 
                                              fmin=fmin, fmax=fmax)

    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

    if plot_flag:

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        librosa.display.specshow(mel_spec_db, x_axis='time',  y_axis='mel', 
                                 sr=sample_rate, hop_length=hop_length, 
                                 fmin=fmin, fmax=fmax, ax=ax)
        title = 'n_mels={},  fmin={},  fmax={}'
        ax.set_title(title.format(n_mels, fmin, fmax))
        plt.show()

    return mel_spec_db

