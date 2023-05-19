# -*- coding: utf-8 -*-
"""

Preprocessing
The first steps to be taken in the classification pipeline. 
In this case preprocessing will consist of two steps:
    1. Endpoint detection
        - leading and trailing silence will be removed from the beginning and 
        end of the signal
    2. Normalization 
        - the magnitude of the signal will be normalized to be between -1 and 1
        this is done to reduce the effect the volume of the speech and microphone
        position

"""

import librosa
import matplotlib.pyplot as plt
import numpy as np

def endpoint_detection(signal, proportion = 0.3):
    avg = np.mean(np.abs(signal))
    thresh = avg*proportion
    abs_signal = np.abs(signal)

    signal_thresh = np.where(abs_signal > thresh)[0]
    begin = signal_thresh[0]
    end = signal_thresh[-1]

    clean_signal = signal[begin:end]
    
    return clean_signal

def normalization(signal):
    X_std = (signal - signal.min()) / (signal.max() - signal.min())
    norm_signal = X_std * (1 + 1) - 1
    return norm_signal


