import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import cwt
from joblib import Parallel, delayed
from natsort import natsorted
import os



n_cpus = os.cpu_count()//2

def wavelet_psd(signal, sampling_rate, freqs, downsampling_factor, wavelet='cmor1.5-1.0'):
    '''
    Calculate the power spectral density of a signal using the continuous wavelet transform. Then downsample to the
    desired frequency resolution.
    :param downsampling_factor: downsampling factor for the wavelet transform, if choose 1, then no downsampling
    if choose 2, then downsample to half the frequency resolution, etc.
    :param signal: 1D numpy array representing the input time-domain signal
    :param sampling_rate: the sampling rate of the signal in Hz
    :param freqs: a list or numpy array of frequencies for which to calculate the PSD
    :param wavelet: the wavelet family to use for the CWT (default: 'cmor' for complex Morlet wavelet)
    :return:
    '''
    def wavelet_parallel(datain, scales, wavelet, sampling_rate):
        cwt_matrix, freq = cwt(datain, scales, wavelet, sampling_period=1 / sampling_rate)
        cwt_matrix_sq = np.abs(cwt_matrix) ** 2
        cwt_matrix_sq_dB = 10 * np.log10(cwt_matrix_sq)
        cwt_matrix_mean = np.mean(cwt_matrix_sq_dB, axis=0)

        # Filter the signal to remove high-frequency noise, choose cutoff at 90 Hz because the frequency of interest
        # is up to 100 Hz
        cutoff_frequency = 90  # Choose the appropriate cutoff frequency (90 Hz)
        filtered_cwt_mean = butter_lowpass_filter(cwt_matrix_mean, cutoff_frequency, sampling_rate)

        # Zero padding the signal so that it can be divided by the downsampling factor
        remainder = len(filtered_cwt_mean) % downsampling_factor
        padding_length = downsampling_factor - remainder if remainder != 0 else 0
        padded_cwt_meanm = np.pad(filtered_cwt_mean, (0, padding_length), mode='constant')

        downsampled_signal = np.mean(padded_cwt_meanm.reshape(-1, downsampling_factor), axis=1)

        return downsampled_signal, freq, cwt_matrix_sq_dB

    # Calculate the scales parameter for the wavelet transform
    scales = sampling_rate * central_frequency(wavelet) / freqs

    # Perform CWT using the Morlet wavelet
    ParallelOutput = Parallel(n_jobs=n_cpus - 1)(delayed(wavelet_parallel)(signal[:,i], scales, wavelet, sampling_rate)
                                             for i in range(0, len(signal.T)))
    ft = ParallelOutput[0][1]
    S1xx = np.array([ParallelOutput[i][0] for i in range(len(ParallelOutput))])
    spectro = np.array([ParallelOutput[i][2] for i in range(len(ParallelOutput))])
    return S1xx, ft, spectro

def choose_wave_wavelet_overtime(brainwave, tepdata, Fs, psdclass, downsample=40):

    '''
    Add and calculate spectrogram for each brainwave to the psdclass
    :param brainwave:
    :param tepdata:
    :param Fs:
    :param psdclass:
    :return:
    '''

    for j in brainwave:
        if j == 'alpha':  # select alpha brainwave
            freqs = np.arange(12, 8, -1)
            teppsd,freq,Spec = wavelet_psd(tepdata, Fs, freqs, downsampling_factor=downsample)
            psdclass.addalpha(teppsd)
            psdclass.addalphaspectrogram(Spec)

        if j == 'beta':  # select beta brainwave
            freqs = np.arange(30, 12, -1)
            teppsd,freq,Spec = wavelet_psd(tepdata, Fs, freqs, downsampling_factor=downsample)
            psdclass.addbeta(teppsd)
            psdclass.addbetaspectrogram(Spec)

        elif j == 'gamma':  # select gamma brainwave
            freqs = np.arange(70, 30, -1)
            teppsd,freq,Spec = wavelet_psd(tepdata, Fs, freqs, downsampling_factor=downsample)
            psdclass.addgamma(teppsd)
            psdclass.addgammaspectrogram(Spec)

        elif j == 'high_gamma':  # select high gamma brainwave
            freqs = np.arange(100, 70, -1)
            teppsd,freq,Spec = wavelet_psd(tepdata, Fs, freqs, downsampling_factor=downsample)
            psdclass.addhgamma(teppsd)
            psdclass.addhgammaspectrogram(Spec)

