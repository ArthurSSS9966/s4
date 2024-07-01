import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dtaidistance import dtw
from joblib import Parallel, delayed
from scipy import spatial
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.signal import butter, correlate, filtfilt, sosfiltfilt
from skimage.measure import block_reduce
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

#########################
# Import from local Library


def simtable(reg1data, reg2data, method, output_path, **kwargs):
    """

    :param reg1data:
    :param reg2data:
    :param method:
    :return:
    """
    if "electrode_locations" in kwargs:
        electrode_locations = kwargs["electrode_locations"]
    else:
        electrode_locations = None
    print("SVD Check Start")
    SVD_data = reg1data[60 * 2000 : 80 * 2000, :]
    SVD_Check(SVD_data, output_path)
    simout = np.zeros((len(reg1data.T), len(reg2data.T)))
    print("Similarity Calculation Start")
    for i in range(0, len(reg1data.T)):
        for k in range(0, len(reg2data.T)):
            simout[i, k] = signal_similarity(
                reg1data[:, i], reg2data[:, k], method
            )
    print("Similarity Calculation Complete")
    # Calculate off-diagonal mean and normalize based on size
    off_diag_mean = np.mean(
        np.abs(simout[np.where(~np.eye(simout.shape[0], dtype=bool))])
    ) / (2 * (np.sqrt(len(reg1data.T) * len(reg2data.T))))
    print("Off-diagonal mean: " + str(off_diag_mean * 100))
    plt.figure()
    plt.pcolormesh(simout, cmap="seismic", vmax=1, vmin=-1)
    plt.colorbar()
    plt.title("Electrode Similarity")
    # Plot horizontal line at the bottom of the plot according to the electrode locations
    if electrode_locations is not None:
        # Create a list to store the x-tick labels
        xtick_labels = []
        # Create a list to store the x-tick positions
        xtick_positions = []

        for i in range(len(electrode_locations)):
            start_location = electrode_locations["Channel"][i][0]
            end_location = electrode_locations["Channel"][i][-1]
            region = electrode_locations["Label"][i]

            # Add a black bar at the top of the plot
            plt.hlines(
                -1, start_location, end_location, color="black", linewidth=5
            )

            # Add the region name to the x-tick labels
            xtick_labels.append(region)

            # Add the middle position of the region to the x-tick positions
            xtick_positions.append((start_location + end_location) / 2)

        # Set the x-tick positions and labels
        plt.xticks(xtick_positions, xtick_labels, rotation=45)

    # Add off-diagonal mean with 3 decimal places
    plt.text(
        0,
        len(reg2data.T) + 0.5,
        "Off-diagonal mean: " + str(round(100 * off_diag_mean, 3)),
    )
    plt.savefig(os.path.join(output_path, "Electrode_Similarity.png"))
    return simout


def SVD_Check(data, output_path):
    """
    Perform SVD on the signal and plot the cumulative sum of the singular values
    :param data:
    :param output_path:
    :return:
    """
    # Normalize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Perform PCA on the dataset
    cpa = PCA(n_components=np.min(np.shape(data))).fit(data)
    pcs = cpa.components_[0, :] ** 2

    # Extract the explained variance ratio
    explained_variance_ratio = cpa.explained_variance_ratio_

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[1].plot(pcs / np.sum(pcs))
    ax[1].set_xlabel("Channel Number", fontsize=18)
    ax[1].set_ylabel("% Contribution to First Feature", fontsize=18)

    # Rearrange the singular values in descending order
    S = explained_variance_ratio
    # Plot the normalized cumulative sum of the singular values
    ax[0].plot(np.cumsum(S) / np.sum(S))
    # Plot the threshold line at 0.95
    ax[0].axhline(y=0.95, color="r", linestyle="--", label="95% Threshold")
    # Plot the vertical line at the number of singular values required to reach 95% variance
    ax[0].axvline(
        x=np.argmax(np.cumsum(S) / np.sum(S) > 0.95),
        color="g",
        linestyle="--",
        label="95% Variance",
    )
    ax[0].set_xlabel("Number of Features", fontsize=18)
    ax[0].set_ylabel("Cumulative Sum of Explained Variance", fontsize=18)
    ax[0].set_title(
        "Normalized Cumulative Sum of Explained Variance", fontsize=18
    )

    plt.savefig(os.path.join(output_path, "SVD.png"))
    # plt.show()


def signal_similarity(x1, x2, method="cs"):
    """
    cs: Cosine Similarity, dtw: dynamic time wrapping, pcc: Pearson correlation coefficient
    :param x1:
    :param x2:
    :param method:
    :return:
    """
    if method == "cs":
        if (np.mean(x1) != 0) and (
            np.mean(x2) != 0
        ):  # To ensure no calculation ERROR happens
            result = 1 - spatial.distance.cosine(x1, x2)
            return result
        return 0
    elif method == "dtw":
        result = dtw.distance_fast(x1, x2, use_pruning=True)
        return result
    elif method == "pcc":
        if (np.mean(x1) != 0) and (np.mean(x2) != 0):
            result = calc_correlation(x1, x2)
            return result
        return 0


def simtablev2(electrode_signal, method="cs"):
    def compute_reduced_similarity_matrix(similarity_matrix, cluster_labels):
        num_clusters = len(set(cluster_labels))
        reduced_matrix = np.zeros((num_clusters, num_clusters))

        for i, label_i in enumerate(set(cluster_labels)):
            indices_i = np.where(cluster_labels == label_i)[0]
            for j, label_j in enumerate(set(cluster_labels)):
                indices_j = np.where(cluster_labels == label_j)[0]
                reduced_matrix[i, j] = np.mean(
                    similarity_matrix[np.ix_(indices_i, indices_j)]
                )

        return reduced_matrix

    if method == "cs":
        # Normalize the electrode signals (if they are not already normalized)
        normalized_electrode_signals = np.array(
            [signal / np.linalg.norm(signal) for signal in electrode_signal.T]
        )

        # Compute the cosine similarity matrix
        cosine_similarity_matrix = cosine_similarity(
            normalized_electrode_signals
        )
    else:
        cosine_similarity_matrix = []

    # Plot the cosine similarity matrix
    # Perform hierarchical clustering using the 'average' method
    clusters = linkage(cosine_similarity_matrix, method="average")

    electrode_labels = np.array(
        ["E" + str(i) for i in range(1, len(electrode_signal.T) + 1)]
    )

    # Create a dendrogram plot
    plt.figure(figsize=(10, 7))
    dendrogram(clusters, labels=electrode_labels)
    plt.title("Electrode Cosine Similarity Dendrogram")
    plt.show()
    # Cut the dendrogram tree to obtain clusters
    max_d = 10  # Adjust this value to control the number of clusters
    cluster_labels = fcluster(clusters, max_d, criterion="distance")

    # Compute the reduced similarity matrix
    reduced_cosine_similarity_matrix = compute_reduced_similarity_matrix(
        cosine_similarity_matrix, cluster_labels
    )

    # Create a heatmap plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        reduced_cosine_similarity_matrix,
        annot=True,
        cmap="coolwarm",
        xticklabels=cluster_labels,
        yticklabels=cluster_labels,
    )
    plt.title("Reduced Electrode Cosine Similarity Heatmap")
    plt.show()


def calc_correlation(actual, predic):
    a_diff = actual - np.mean(actual)
    p_diff = predic - np.mean(predic)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff**2)) * np.sqrt(np.sum(p_diff**2))
    return numerator / denominator


def bad_ele_similarity_based(similarity):
    badelectrode = []
    for i in range(0, len(similarity)):
        for j in range(0, np.shape(similarity)[1]):
            if (i != j) and (similarity[i][j] > 0.8):
                badelectrode.append([i, j])
    badelectrode = np.unique(badelectrode)
    return badelectrode


def cal_correlation(psddata, phasel, phase, freq, method="cs"):
    size = len(psddata) - 10
    elesize = np.shape(psddata[0].Sxx)[0]
    correlation_mean = 0
    for j in range(0, elesize):
        correlation = np.zeros((size))
        count = 0
        for i in range(0, size):
            if psddata[i].phase == phasel and psddata[i].seqid == phase:
                movedata = psddata[i].movedata
                if freq == "alphapsd":
                    psd = psddata[i].alphapsd[j, :]
                if freq == "betapsd":
                    psd = psddata[i].betapsd[j, :]
                if freq == "gammapsd":
                    psd = psddata[i].gammapsd[j, :]
                if freq == "highgammapsd":
                    psd = psddata[i].highgammapsd[j, :]
                movedata = block_reduce(
                    movedata, int(len(movedata) / (len(psd))), func=np.mean
                )
                movedata = movedata[0 : len(psd)]
                correlation[i] = signal_similarity(movedata, psd, method)
                print(correlation[i])
                if abs(correlation[i]) > 0.75:
                    plt.plot(
                        (movedata - min(movedata))
                        / (max(movedata) - min(movedata)),
                        color="black",
                    )
                    plt.plot(
                        (psd - min(psd)) / (max(psd) - min(psd)), color="red"
                    )
                count += 1
        plt.title(
            "Movement Data With Spectrogram During" + phasel + " in " + freq
        )
        plt.show()
        correlation_mean = np.sum(correlation) / count
        print("mean Correlation: ", correlation_mean)
    return correlation_mean


def self_correlation(data_in):
    N = len(data_in)

    time_lag = 1 * 2000  # 1sec * 2000/sec

    # Compute auto-correlation
    corr_out = correlate(data_in, data_in, mode="same")
    corr_out /= np.max(corr_out)
    return corr_out[(N + 1) // 2 : (N + 1) // 2 + time_lag]


def snr_calculation(eeg_signal, fs=2000):
    def get_power(signal):
        power_signal = np.sum(signal**2) / len(signal)
        return power_signal

    def get_quality_signal(signal, fs):
        sub_bands = [(13, 30), (30, 50), (50, 100), (100, 200), (200, 350)]
        power_quality_signal = 0
        filter_order = 5
        for low, high in sub_bands:
            quality_signal = butter_bandpass_filter(
                signal, low, high, fs, filter_order
            )
            power_sub_band = get_power(quality_signal)
            freq_norm = np.sqrt(
                low * high
            )  # Geometric mean of sub-band frequencies
            normalized_power = (
                power_sub_band / freq_norm
            )  # Normalize power by freq_norm
            power_quality_signal += normalized_power
        return power_quality_signal

    def get_noisy_signal(signal, fs):
        filter_order = 5
        # noise_floor = estimate_noise_floor(signal, fs)
        noisy_signal = butter_bandpass_filter(
            signal, 350, fs // 3, fs, filter_order
        )
        power_noisy_signal = get_power(noisy_signal)
        freq_norm = np.sqrt(
            350 * (fs // 3)
        )  # Geometric mean of noisy sub-band frequencies
        normalized_power = (
            power_noisy_signal / freq_norm
        )  # Normalize power by freq_norm
        return normalized_power

    def estimate_noise_floor(signal, fs, window_size=1000):
        noise_floor = 0
        num_windows = len(signal) // window_size
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window_signal = signal[start:end]
            noise_floor += get_power(window_signal)
        noise_floor /= num_windows
        return noise_floor

    power_quality_signal = get_quality_signal(eeg_signal, fs)
    power_noisy_signal = get_noisy_signal(eeg_signal, fs)

    if power_noisy_signal <= 0:
        raise ValueError(
            "Noisy signal power is non-positive. Please check the noise floor"
            " estimation."
        )

    # Calculate the SNR
    snr = 10 * np.log10(power_quality_signal / power_noisy_signal)
    return snr


def signal_quality_test(data, output_dir):
    N, E = np.shape(data)

    def plot_correlation(corr_in, lag, output_dir):
        corr_mean = np.mean(corr_in, axis=0)
        plt.figure()

        for i in range(len(corr_in)):
            # Plot the auto-correlation
            plt.plot(lag, corr_in[i, :], color="gray", alpha=0.2)

        plt.plot(lag, corr_mean, color="k")
        plt.ylim([-0.3, 0.5])
        plt.xlabel("Lag (s)")
        plt.ylabel("Auto-correlation")
        plt.title("Auto-correlation of sEEG Signal Data")
        plt.savefig(output_dir + "Auto-correlation of sEEG Signal Data.png")
        # plt.show()
        # plt.close()

    def plot_SNR(SNR, output_dir):
        plt.figure()
        plt.plot(SNR)
        plt.xlabel("Channel")
        plt.ylabel("SNR")
        plt.title("SNR of each channel")
        plt.savefig(output_dir + "SNR of sEEG Signal Data.png")
        # plt.show()
        # plt.close()

    n_cpus = os.cpu_count() // 2
    with Pool(n_cpus) as pool:
        corrMat = pool.map(self_correlation, data.T)
        SNR = pool.map(snr_calculation, data.T)

    corrMat = np.array(corrMat)
    SNRMat = np.array(SNR)

    time_lag = 1 * 2000
    lag = np.linspace(0, 1, time_lag)
    plot_correlation(corrMat, lag, output_dir)
    plot_SNR(SNRMat, output_dir)

    return corrMat, SNRMat


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


if __name__ == "__main__":
    signal = np.random.rand(10000, 20)
