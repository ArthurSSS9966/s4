from scipy.signal import butter, hilbert, filtfilt
import numpy as np

def cal_hilbert(data, Fs, bandpass=[]):
    '''

    :param data:
    :param Fs:
    :param bandpass:
    :return:
    '''

    def average_over_time(data):
        '''

        :param data: Must be a (20*N)*E electrode data
        :return:
        '''
        n_samples_per_segment = 50

        # Reshape the data array into segments
        n_segments = data.shape[0] // n_samples_per_segment
        data_reshaped = data[:n_segments * n_samples_per_segment].reshape(n_segments, n_samples_per_segment,
                                                                          data.shape[1])

        # Compute the mean of each segment
        data_mean = np.mean(data_reshaped, axis=1)

        return data_mean

    nyquist_freq = Fs * 0.5
    beta_band_norm = np.array(bandpass) / nyquist_freq
    b, a = butter(N=4, Wn=beta_band_norm, btype='band')
    seeg_data_filt = filtfilt(b, a, data, axis=0)

    seeg_hilbert = hilbert(seeg_data_filt)
    seeg_amp = np.abs(seeg_hilbert)
    seeg_phase = np.angle(seeg_hilbert)

    seeg_power = seeg_amp ** 2
    seeg_power = 10 * np.log10(seeg_power)

    seeg_power = average_over_time(seeg_power)
    seeg_phase = average_over_time(seeg_phase)

    return seeg_power, seeg_phase


def choose_wave_hilbert(brainwave, data, Fs, psdclass):
    for j in brainwave:
        if j == 'alpha':  # select alpha brainwave
            fband = [8, 12]
            teppsd, tepphase = cal_hilbert(data, Fs, fband)
            psdclass.addalpha(teppsd.T[:, 1:-2])
            psdclass.addangle(tepphase)
        if j == 'beta':  # select beta brainwave
            fband = [13, 30]
            teppsd, tepphase = cal_hilbert(data, Fs, fband)
            psdclass.addbeta(teppsd.T[:, 1:-2])
            psdclass.addangle(tepphase)
        elif j == 'gamma':  # select gamma brainwave
            fband = [31, 70]
            teppsd, tepphase = cal_hilbert(data, Fs, fband)
            psdclass.addgamma(teppsd.T[:, 1:-2])
            psdclass.addangle(tepphase)
        elif j == 'high_gamma':  # select high gamma brainwave
            fband = [70, 100]
            teppsd, tepphase = cal_hilbert(data, Fs, fband)
            psdclass.addhgamma(teppsd.T[:, 1:-2])
            psdclass.addangle(tepphase)