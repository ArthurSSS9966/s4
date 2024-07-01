
from multitaper import MTSpec
from joblib import Parallel, delayed
import numpy as np
import os

n_cpus = os.cpu_count() // 2

def cal_psd(totdata, Fs, n_segment=1):
    '''
    Calculate PSD for entire dataset
    :param totdata:
    :param Fs:
    :param minseg:
    :param method:
    :return:
    '''
    def multitaper_parallel_segmented(datain, Fs, n_segments):
        # Very necessary, defines the frequency resolution
        df = 0.25
        nfft = Fs / df

        # Split the data into segments
        segment_base_len, remainder = divmod(len(datain), n_segments)
        segment_lengths = [segment_base_len] * n_segments
        # Add reminder to the last segment
        segment_lengths[-1] += remainder

        # Calculate the PSD for each segment
        Sxx_segments = []
        start = 0

        for segment_len in segment_lengths:
            segment_data = datain[start:start + segment_len]
            MTclass = MTSpec(segment_data, dt=1 / Fs, nw=3.5, kspec=5, nfft=nfft)
            ft, Sxx = MTclass.rspec()
            Sxx_segments.append(Sxx)
            start += segment_len

        # Average the PSDs from all segments
        Sxx_avg = np.mean(Sxx_segments, axis=0)

        return ft, Sxx_avg

    ParallelOutput = Parallel(n_jobs=n_cpus)(delayed(multitaper_parallel_segmented)(totdata[:, i], Fs, n_segment)
                                             for i in range(0, len(totdata.T)))
    S1xx = np.array([ParallelOutput[i][1] for i in range(len(ParallelOutput))])
    S1xx = np.squeeze(S1xx, axis=-1)
    S1xx = 10 * np.log10(S1xx)
    ft = ParallelOutput[0][0]

    ft = ft.flatten()
    teppsd = S1xx[:, (ft > 0) & (ft <= 200)]
    ft = ft[(ft > 0) & (ft <= 200)]
    return teppsd, ft