
import numpy as np
from numpy import matlib

def detrend_data(totdata):
    '''
    General function to detrend the input data
    :param totdata:
    :return:
    '''
    tepmean = np.median(totdata, axis=0)
    tepmean = matlib.repmat(tepmean, len(totdata), 1)
    totdata = totdata - tepmean
    return totdata


def choose_wave_spec(brainwave, tepdata, f, psdclass):
    for j in brainwave:
        if j == 'alpha':  # select alpha brainwave
            teppsd = tepdata[:, (f >= 8) & (f <= 12), :]
            psdclass.addalphaspectrogram(teppsd)
        if j == 'beta':  # select beta brainwave
            teppsd = tepdata[:, (f >= 12) & (f <= 30), :]
            psdclass.addbetaspectrogram(teppsd)
        elif j == 'gamma':  # select gamma brainwave
            teppsd = tepdata[:, (f >= 30) & (f < 70), :]
            psdclass.addgammaspectrogram(teppsd)
        elif j == 'high_gamma':  # select high gamma brainwave
            teppsd = tepdata[:, (f >= 70) & (f < 100), :]
            psdclass.addhgammaspectrogram(teppsd)
    return psdclass


def choose_wave_psd_range(brainwave, tepdata, f, psdclass):
    for j in brainwave:
        if j == 'alpha':  # select alpha brainwave
            teppsd = tepdata[:, (f >= 8) & (f <= 12)]
            psdclass.addalpha(teppsd)
        if j == 'beta':  # select beta brainwave
            teppsd = tepdata[:, (f >= 13) & (f <= 30)]
            psdclass.addbeta(teppsd)
        elif j == 'gamma':  # select gamma brainwave
            teppsd = tepdata[:, (f >= 31) & (f < 70)]
            psdclass.addgamma(teppsd)
        elif j == 'high_gamma':  # select high gamma brainwave
            teppsd = tepdata[:, (f >= 71) & (f < 100)]
            psdclass.addhgamma(teppsd)
    return psdclass

