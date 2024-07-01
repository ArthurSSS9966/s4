import os

import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import welch
from scipy.stats import shapiro
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from rm_artifacts import cal_avg

# Import from local library
from utilfun import band_to_value, bootstrap, convert_pvalue_to_asterisks


def plot_raw(x, task, tasklabel, electrode, Fs=2000):
    tasktime = np.array(task[0, :-2], task[1, 0])
    data = x[int(tasktime[0] - Fs * 0.2) : int(tasktime[-1] + Fs * 1), :]
    plttime = np.linspace(
        tasktime[0] - Fs * 0.2, tasktime[-1] + Fs * 1, len(data)
    )
    plttime = plttime / Fs
    avg_data = cal_avg(data, electrode)
    fig, axs = plt.subplots(len(electrode))
    fig.suptitle("Plot of raw data")
    for i in range(0, len(electrode)):
        axs[i].plot(plttime, avg_data[:, i])

        for k in range(0, len(tasktime)):
            axs[i].axvline(
                tasktime[k] / Fs,
                label=tasklabel[k],
            )

    plt.show()


def plot_histo(PSD):
    Betadis = np.zeros((0))
    for i in range(0, len(PSD)):
        Betadis = np.hstack((Betadis, PSD[i][7, :]))
    plt.hist(Betadis)
    plt.show()
    shapiro(Betadis)


def plot_heat(
    trdata, trid, OUT_DIR, type, phasesel, region, reqband, vmax=15, vmin=-25
):
    """
    Plot and save heat map of 2D matrix trdata, with each row correspond to the seqid
    :param trdata:
    :param trid:
    :param OUT_DIR:
    :param type:
    :param phasesel:
    :param region:
    :param reqband:
    :param vmax:
    :param vmin:
    :return:
    """
    savename = OUT_DIR + type + "Latent_HEATMAP.png"
    sortedid = np.argsort(np.array(trid))  # Sort list based on seqid
    seqid = 1
    picprow = 10
    for i in range(0, len(sortedid)):
        if trid[sortedid[i]] == seqid:
            picind = i % 10 + (seqid - 1) * picprow + 1
            ax1 = plt.subplot(len(np.unique(trid)), picprow, picind)
            ax1.pcolormesh(
                trdata[sortedid[i]][0:, :], cmap="seismic", vmax=vmax, vmin=vmin
            )
        else:
            seqid += 1
            picind = (seqid - 1) * picprow + i % 10 + 1
            ax1 = plt.subplot(len(np.unique(trid)), picprow, picind)
            ax1.pcolormesh(
                trdata[sortedid[i]][0:, :], cmap="seismic", vmax=vmax, vmin=vmin
            )
    plt.suptitle(
        type
        + " heat map in "
        + region
        + " during "
        + phasesel
        + " for "
        + reqband
    )
    plt.savefig(savename)
    plt.show()


def plot_mean_Channel(x1, x2, region1, region2, Fs=2000):
    t = np.arange(0, len(x1) / Fs, 1 / Fs)
    fig, axs = plt.subplots(2)
    fig.suptitle("Raw data between " + region1 + " and " + region2)
    axs[0].plot(t, x1)
    axs[0].set_title(region1)
    axs[1].plot(t, x2)
    axs[1].set_title(region2)
    for ax in axs.flat:
        ax.set(xlabel="time(s)", ylabel="amplitude (uV)")
    fig.show()


def plot_similarity(simout, patient, phase, task, region):
    plt.pcolormesh(simout, cmap="seismic", vmax=1, vmin=-1)
    plt.title(
        "Electrode Similarity Heat map in "
        + region
        + " during "
        + phase
        + " phase for "
        + task
        + " in patient "
        + patient
    )
    plt.show()


def plot_move(psdclass, condition, phasesel):
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    for i in range(0, len(psdclass)):
        if psdclass[i].phase == phasesel:
            plt.plot(psdclass[i].movedata, color=colors[psdclass[i].seqid])
    plt.title(
        condition
        + " Hand Horizontal Movement Velocity (mm/s) During "
        + phasesel
    )
    plt.ylim((-250, 200))
    plt.show()


def plot_class_score(cont_out, pred_out, negative_out):
    """

    :param cont_out:
    :param pred_out:
    :param negative_out:
    :return:
    """
    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle("Classification Score")
    axs[0].hist(cont_out.flatten())
    axs[1].hist(pred_out.flatten())
    axs[2].hist(negative_out.flatten())
    for ax in axs.flat:
        ax.set(xlabel="Classification Score")
    for ax in axs.flat:
        ax.label_outer()
    fig.show()


def plot_beta_hist(psddata, phaselabel, channel, output_dir, freqband="beta"):
    plotcolors = [
        "blue",
        "yellow",
        "red",
        "purple",
        "green",
        "cyan",
        "navy",
        "royalblue",
        "teal",
        "lime",
        "olive",
        "gold",
        "orange",
        "coral",
        "salmon",
        "indianred",
    ]
    plt.figure()
    for p in range(len(phaselabel)):
        phase = phaselabel[p]
        psdbeta = None
        for i in range(len(psddata)):
            if psddata[i].phase == phase:
                if psdbeta is None:
                    psdbeta = psddata[i].getband(freqband)[channel, :]
                else:
                    psdbeta = np.vstack(
                        (psdbeta, psddata[i].getband(freqband)[channel, :])
                    )
        psdbeta = 10 * np.log10(psdbeta)
        psdmean = np.median(psdbeta, axis=-1)
        plt.hist(
            psdmean,
            label=phaselabel[p],
            color=plotcolors[p],
            alpha=0.5,
            bins=20,
        )
        plt.axvline(np.median(psdmean), color=plotcolors[p])
    plt.legend()
    plt.title(
        "Channel: " + str(channel) + " PSD Distribution within " + freqband
    )
    plt.savefig(output_dir + "Channel" + str(channel) + "_hist.png")
    # plt.show()


def plot_beta_hist_gng(
    psddata, phaselabel, channel, output_dir, gng, freqband="beta"
):
    plotcolors = [
        "orange",
        "blue",
        "green",
        "red",
        "magenta",
        "cyan",
        "navy",
        "royalblue",
        "teal",
        "lime",
        "olive",
        "gold",
        "orange",
        "coral",
        "salmon",
        "indianred",
    ]
    for p in range(len(phaselabel)):
        phase = phaselabel[p]
        psdbeta = None
        if phaselabel[p] == "Response":
            psdbeta_no_go = None
            for i in range(len(psddata)):
                trial = psddata[i].trial
                if psddata[i].phase == phase:
                    if gng[trial] == 1:
                        if psdbeta is None:
                            psdbeta = psddata[i].getband(freqband)[channel, :]
                        else:
                            psdbeta = np.vstack(
                                (
                                    psdbeta,
                                    psddata[i].getband(freqband)[channel, :],
                                )
                            )
                    else:
                        if psdbeta_no_go is None:
                            psdbeta_no_go = psddata[i].getband(freqband)[
                                channel, :
                            ]
                        else:
                            psdbeta_no_go = np.vstack(
                                (
                                    psdbeta_no_go,
                                    psddata[i].getband(freqband)[channel, :],
                                )
                            )

            psdmean = np.mean(psdbeta, axis=-1)
            psdmean_no_go = np.mean(psdbeta_no_go, axis=-1)
            plt.hist(
                psdmean,
                label=phaselabel[p],
                color=plotcolors[p],
                alpha=0.5,
                bins=20,
            )
            plt.hist(
                psdmean_no_go,
                label="Response_No_go",
                color=plotcolors[p + 1],
                alpha=0.5,
                bins=20,
            )

            plt.xlim([0, 35])
            plt.axvline(np.mean(psdmean), color=plotcolors[p])
            plt.axvline(np.mean(psdmean_no_go), color=plotcolors[p + 1])

        else:
            for i in range(len(psddata)):
                if psddata[i].phase == phase:
                    if psdbeta is None:
                        psdbeta = psddata[i].getband(freqband)[channel, :]
                    else:
                        psdbeta = np.vstack(
                            (psdbeta, psddata[i].getband(freqband)[channel, :])
                        )

            psdmean = np.mean(psdbeta, axis=-1)
            plt.hist(
                psdmean,
                label=phaselabel[p],
                color=plotcolors[p],
                alpha=0.5,
                bins=20,
            )
            plt.axvline(np.mean(psdmean), color=plotcolors[p])
    plt.legend()
    plt.xlabel("PSD (dB)")
    plt.title(
        "Channel: " + str(channel) + " PSD Distribution within " + freqband
    )
    plt.savefig(output_dir + "Channel_" + str(channel) + "_hist.png")
    plt.show()


def add_sig_star(groups, p_values, ax=None, violin_parts=None):
    def find_y_position(x_index, violin_parts):
        y_max = 0
        for vp in violin_parts:
            for path in vp.get_paths():
                vertices = path.vertices
                x_values = vertices[:, 0]
                y_values = vertices[:, 1]
                filtered_y_values = y_values[
                    (x_values >= x_index) & (x_values <= x_index + 0.2)
                ]
                y_max_current = (
                    0
                    if filtered_y_values.size == 0
                    else np.max(filtered_y_values)
                )
                y_max = max(y_max, y_max_current)
        return y_max

    if ax is None:
        ax = plt.gca()

    y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ylim_upper = ax.get_ylim()[1]

    occupied_positions = []

    for i, ((x1, x2), p_value) in enumerate(zip(groups, p_values)):
        y_start = max(
            find_y_position(x1, violin_parts), find_y_position(x2, violin_parts)
        )
        y_line = y_start

        while any(
            (x1 < pos[1] and x2 > pos[0])
            for pos in occupied_positions
            if abs(pos[2] - y_line) < y_offset * 2.3
        ):
            y_line += y_offset * 1.5
        occupied_positions.append((x1, x2, y_line))

        ax.plot(
            [x1, x1, x2, x2],
            [y_line, y_line + y_offset, y_line + y_offset, y_line],
            lw=1.5,
            c="black",
        )

        stars = convert_pvalue_to_asterisks(p_value)
        ax.text(
            (x1 + x2) * 0.5,
            y_line + y_offset,
            stars,
            ha="center",
            va="bottom",
            backgroundcolor="none",
        )

        ylim_upper = max(ylim_upper, y_line + y_offset * 4)

    ax.set_ylim(ax.get_ylim()[0], ylim_upper)


def _plot_single_channel_spectrogram(spectrogram, channel, freq_plot, **kwargs):
    """

    :param spectrogram: 3D array with [channel, freq_bin, time]
    :param channel:
    :param band:
    :param freq_plot: 1D vector of frequency bin value
    :return:
    """
    # Plot the spectrogram
    if "movement_type" in kwargs:
        movement_type = kwargs["movement_type"]
        movement = kwargs["movement"]
        print("Movement type detected: " + str(movement_type))
    else:
        movement_type = "Uni"
        movement = []

    if movement != []:
        number_rows = 3
        height_ratios = [6, 1, 1]
    else:
        number_rows = 2
        height_ratios = [6, 1]

    plot_data = spectrogram[channel, :, :]
    time = np.linspace(
        start=0, stop=plot_data.shape[1] / 2000, num=plot_data.shape[1]
    )
    # Normalize the data for each row to have zero mean and unit variance
    plot_data = (
        plot_data - np.mean(plot_data, axis=1, keepdims=True)
    ) / np.std(plot_data, axis=1, keepdims=True)
    # Get the 1 and 99 percentile of the data to set as the plot limit
    vmin = np.percentile(plot_data, 1)
    vmax = np.percentile(plot_data, 99)

    # Create 2-row subplot grid
    fig = plt.figure(figsize=(12, 10))

    gs = gridspec.GridSpec(
        number_rows, 1, height_ratios=height_ratios
    )  # Set height ratios

    # First axes object for the spectrogram
    ax = plt.subplot(gs[0])
    im = ax.imshow(
        plot_data,
        cmap="jet",
        aspect="auto",
        origin="lower",
        extent=[time[0], time[-1], freq_plot[0], freq_plot[-1]],
    )
    ax.set_title(f"Channel {channel}, Spectrogram")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    # Set the colorbar and the range from vmin to vmax at the bottom of the plot
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.2)
    cbar.set_label("Power Spectral Density (V^2)")
    im.set_clim(vmin, vmax)
    ax.set_title(f"Channel {channel} Spectrogram, {movement_type} movement")


def plot_spectrogram_all(
    spectrogram, Fs, task, movement, taskname, title, save_path, **kwargs
):
    """
    Plot the spectrogram
    :param spectrogram:
    :param raw_data:
    :param Fs:
    :param task:
    :param movement:
    :param taskname:
    :param title:
    :param save_path:
    :return:
    """
    # Plot the spectrogram
    if "movement_type" in kwargs:
        movement_type = kwargs["movement_type"]
        print("Movement type detected: " + str(movement_type))
    else:
        movement_type = None
    if "phase_name" in kwargs:
        phase_name = kwargs["phase_name"]
        phase_ind = taskname.index(phase_name)
    else:
        phase_name = "All"
        phase_ind = 0

    for i in tqdm(range(len(task) - 1), desc="Plotting spectrogram"):
        alpha_data = spectrogram[i].alphaspec
        beta_data = spectrogram[i].betaspec
        gamma_data = spectrogram[i].gammaspec
        high_gamma_data = spectrogram[i].highgammaspec
        movement_type_ind = spectrogram[i].seqid
        if movement_type is not None:
            movement_dir = movement_type[movement_type_ind]
        else:
            movement_dir = "Uni"

        # Convert to power magnitude just for plotting
        alpha_data = 10 ** (alpha_data / 10)
        beta_data = 10 ** (beta_data / 10)
        gamma_data = 10 ** (gamma_data / 10)
        high_gamma_data = 10 ** (high_gamma_data / 10)

        raw_trial = spectrogram[i].rawdata
        # Stack the data along second axis
        spec_data = np.concatenate(
            (alpha_data, beta_data, gamma_data, high_gamma_data), axis=1
        )
        if movement != []:
            hand_movement_data = movement[task[i, 0] : task[i + 1, 0], :]
            number_rows = 4
            height_ratios = [6, 1, 1, 1]
        else:
            number_rows = 3
            height_ratios = [6, 1, 1]
        freq = spectrogram[i].ft

        freq = freq[(freq >= 8) & (freq <= 100)]

        for band in range(5):
            if band == 0:
                band_data = beta_data
                freq_plot = freq[(freq >= 13) & (freq <= 30)]
                band_name = "beta"
                low_pass = 30
                high_pass = 13
            elif band == 1:
                band_data = gamma_data
                freq_plot = freq[(freq >= 30) & (freq <= 70)]
                band_name = "gamma"
                low_pass = 70
                high_pass = 30
            elif band == 2:
                band_data = high_gamma_data
                freq_plot = freq[(freq >= 70) & (freq <= 100)]
                band_name = "high_gamma"
                low_pass = 100
                high_pass = 70
            elif band == 3:
                band_data = alpha_data
                freq_plot = freq[(freq >= 8) & (freq <= 13)]
                band_name = "alpha"
                low_pass = 13
                high_pass = 8
            else:
                band_data = spec_data
                freq_plot = freq
                band_name = "All"
                low_pass = 100
                high_pass = 8

            time = spectrogram[i].t[: np.shape(band_data)[2] - 1]
            phasetime = task[i][:-2]
            phasetime = (phasetime - phasetime[0]) / Fs

            if "channels" in kwargs:
                channels = kwargs["channels"]
            else:
                channels = range(band_data.shape[0])

            for channel in channels:
                raw_plot_data = raw_trial[:, channel]
                # raw_plot_data = butter_bandpass_filter(raw_plot_data, low_pass, high_pass, Fs, order=5)
                plot_data = band_data[channel, :, :]
                # Normalize the data for each row to have zero mean and unit variance
                plot_data = (
                    plot_data - np.mean(plot_data, axis=1, keepdims=True)
                ) / np.std(plot_data, axis=1, keepdims=True)
                # Get the 1 and 99 percentile of the data to set as the plot limit
                vmin = np.percentile(plot_data, 3)
                vmax = np.percentile(plot_data, 97)

                # Create 2-row subplot grid
                fig = plt.figure(figsize=(12, 10))

                gs = gridspec.GridSpec(
                    number_rows, 1, height_ratios=height_ratios
                )  # Set height ratios

                # First axes object for the spectrogram
                ax = plt.subplot(gs[0])
                im = ax.imshow(
                    plot_data,
                    cmap="jet",
                    aspect="auto",
                    origin="lower",
                    extent=[time[0], time[-1], freq_plot[0], freq_plot[-1]],
                )
                ax.set_ylabel("Frequency (Hz)", fontsize=15)
                # Set the colorbar and the range from vmin to vmax at the bottom of the plot
                cbar = fig.colorbar(
                    im, ax=ax, orientation="horizontal", pad=0.2
                )
                cbar.set_label("Power Spectral Density (V^2)", fontsize=15)
                im.set_clim(vmin, vmax)
                ax.set_title(
                    f"Channel {channel}, trial {i + 1}, {phase_name},"
                    f" Spectrogram, {band_name} band, {movement_dir} movement",
                    fontsize=18,
                )

                # Plot a vertical red line to indicate the phase time, and indicate the name of the phase
                if phase_ind == 0:
                    for j in range(1, len(phasetime)):
                        ax.axvline(
                            x=phasetime[j],
                            color="r",
                            linestyle="--",
                            linewidth=1.5,
                        )
                        # Set the text at the top of the plot
                        # Convert from data coordinates to axis coordinates
                        x_in_axis_coords, _ = ax.transLimits.transform(
                            (phasetime[j], 0)
                        )
                        # Set the text at the top of the plot
                        ax.text(
                            x_in_axis_coords,
                            0.9,
                            taskname[j],
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=ax.transAxes,
                            color="r",
                            fontsize=15,
                        )
                if movement != []:
                    # Second axes object for the 1D hand movement plot
                    ax2 = plt.subplot(gs[3])
                    timemovement = np.linspace(
                        start=time[0],
                        stop=time[-1],
                        num=len(hand_movement_data[:, 0]),
                    )
                    for movedata in hand_movement_data.T:
                        ax2.plot(timemovement, movedata)
                        ax2.set_xlim([timemovement[0], timemovement[-1]])
                        ax2.set_ylabel("Hand Velocity (mm/s)", fontsize=15)
                if phase_ind == 0 and movement != []:
                    for j in range(1, len(phasetime)):
                        ax2.axvline(
                            x=phasetime[j],
                            color="r",
                            linestyle="--",
                            linewidth=1.5,
                        )
                        # Set the text at the top of the plot
                        x_in_axis_coords, _ = ax2.transLimits.transform(
                            (phasetime[j], 0)
                        )
                        # Set the text at the top of the plot
                        ax2.text(
                            x_in_axis_coords,
                            0.9,
                            taskname[j],
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=ax2.transAxes,
                            color="r",
                            fontsize=15,
                        )
                    ax2.set_ylim([-100, 100])
                    # Set the title of the plot
                    ax2.set_title(
                        f"Channel {channel}, trial {i + 1} Hand Movement"
                    )

                # Third axes object for the average power spectral density overtime plot
                ax3 = plt.subplot(gs[1])
                # Calculate the average power spectral density overtime
                avg_psd = np.mean(plot_data, axis=0)
                psd_time = np.linspace(
                    start=time[0], stop=time[-1], num=len(avg_psd)
                )
                # Plot the average power spectral density overtime
                ax3.plot(psd_time, avg_psd)
                ax3.set_xlim([psd_time[0], psd_time[-1]])
                ax3.set_ylabel("Average PSD (V^2)", fontsize=15)
                if phase_ind == 0:
                    for j in range(1, len(phasetime)):
                        ax3.axvline(
                            x=phasetime[j],
                            color="r",
                            linestyle="--",
                            linewidth=1.5,
                        )
                        # Set the text at the top of the plot
                        x_in_axis_coords, _ = ax3.transLimits.transform(
                            (phasetime[j], 0)
                        )
                        # Set the text at the top of the plot
                        ax3.text(
                            x_in_axis_coords,
                            0.9,
                            taskname[j],
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=ax3.transAxes,
                            color="r",
                            fontsize=15,
                        )
                    # Set the title of the plot
                    ax3.set_title(
                        f"Channel {channel}, trial {i + 1} Average Power"
                        " Spectral Density"
                    )

                # Fourth axes object for the raw data plot
                ax4 = plt.subplot(gs[2])
                # Plot the raw data
                raw_time = np.linspace(
                    start=time[0], stop=time[-1], num=len(raw_plot_data)
                )
                ax4.plot(raw_time, raw_plot_data)
                ax4.set_xlim([raw_time[0], raw_time[-1]])
                ax4.set_xlabel("Time (s)", fontsize=15)
                ax4.set_ylabel("Raw Data(uV)", fontsize=15)
                if phase_ind == 0:
                    for j in range(1, len(phasetime)):
                        ax4.axvline(
                            x=phasetime[j],
                            color="r",
                            linestyle="--",
                            linewidth=1.5,
                        )
                        # Set the text at the top of the plot
                        x_in_axis_coords, _ = ax4.transLimits.transform(
                            (phasetime[j], 0)
                        )
                        # Set the text at the top of the plot
                        ax4.text(
                            x_in_axis_coords,
                            0.9,
                            taskname[j],
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=ax4.transAxes,
                            color="r",
                            fontsize=15,
                        )
                    # Set the title of the plot
                    ax4.set_title(f"Channel {channel}, trial {i + 1} Raw Data")

                # Save the plot
                save_path_tep = save_path + f"_channel_{channel}"
                # Create the directory if it does not exist
                if not os.path.exists(save_path_tep + f"\\{band_name}"):
                    os.makedirs(save_path_tep + f"\\{band_name}")
                plt.savefig(
                    save_path_tep
                    + f"\\{band_name}\\_{title}_trial{i + 1}_"
                    f"spectrogram_{band_name}_phase{phase_name}.png"
                )
                plt.close()

    print("Spectrogram and hand movement plot complete")


def plot_violin_hist_half(data1, data2, p_val, ax, position, **kwargs):
    if "electrode_name" in kwargs:
        electrode_name = kwargs["electrode_name"]
    else:
        electrode_name = "electrode" + str(position / 2)

    if "labels" in kwargs:
        labels = kwargs["labels"]
    else:
        labels = ["Fix", "Delay"]

    # Create positions for violin plots
    positions = [position, position + 1]
    colors = ["orange", "purple"]

    # Loop over data1 and data2
    for i, data in enumerate([data1, data2]):
        # Create violin plot
        violin_parts = ax.violinplot(
            data, positions=[positions[i]], showextrema=False, showmedians=False
        )

        # Set color for violin plot
        for vp in violin_parts["bodies"]:
            vp.set_facecolor(colors[i])
            vp.set_alpha(0.5)

            # Make it a half violin plot
            m = np.mean(vp.get_paths()[0].vertices[:, 0])
            vp.get_paths()[0].vertices[:, 0] = np.clip(
                vp.get_paths()[0].vertices[:, 0], m, np.inf
            )

            # Find the interquartile range within the violin plot's path
            quartile1 = np.percentile(data, 25)
            quartile3 = np.percentile(data, 75)
            path = vp.get_paths()[0].vertices
            # Get the indices of the path that correspond to the quartile range
            quartile_mask = (path[:, 1] >= quartile1) & (
                path[:, 1] <= quartile3
            )
            # Color the interquartile range area
            ax.fill_betweenx(
                path[quartile_mask, 1],
                m,
                path[quartile_mask, 0],
                color="black",
                alpha=0.4,
            )

        # Plot median dot
        ax.plot(
            positions[i],
            np.median(data),
            color="white",
            marker="o",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # Histogram
        hist_data, bin_edges = np.histogram(data, bins=20)
        hist_data = hist_data / hist_data.max() * 0.45
        # Calculate the width of each bin
        bin_width = (
            bin_edges[1] - bin_edges[0]
        ) * 0.5  # Adjust the multiplier to control the bin thickness

        # Draw the bars with spacing
        for j in range(len(hist_data)):
            ax.barh(
                bin_edges[:-1][j],
                -hist_data[j],
                left=positions[i],
                height=bin_width,
                align="edge",
                color=colors[i],
                edgecolor="white",
            )

    stars = convert_pvalue_to_asterisks(p_val)
    # Add asterisks and a line between two violin plots
    ax.text(
        positions[0] + 0.5,
        np.max(np.concatenate((data1, data2))) * 1.05,
        stars,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=20,
    )
    ax.plot(
        [positions[0] + 0.1, positions[1] - 0.1],
        [np.max(np.concatenate((data1, data2))) * 1.025] * 2,
        color="black",
        lw=2,
    )

    ax.set_ylim(
        [
            np.min(np.concatenate((data1, data2))) * 0.9,
            np.max(np.concatenate((data1, data2))) * 1.1,
        ]
    )

    # At the end of the function, add the legend.
    if labels is not None and len(labels) == 2:
        custom_lines = [
            Line2D([0], [0], color="orange", lw=4),
            Line2D([0], [0], color="purple", lw=4),
        ]
        ax.legend(custom_lines, labels)

    return positions


def plot_accuracy_map(accuracy_map, band, data_type, title=None):
    N_channels = accuracy_map.shape[0]
    # Plot the accuracy map
    fig_width = len(band) * 3  # each column twice as wide
    fig_height = N_channels // 3  # each row 1/3 times as tall
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(accuracy_map)
    # Set the x ticks
    plt.xticks(np.arange(len(band)), band, rotation=90, fontsize=25)
    # Change the fontsize of the y ticks
    plt.yticks(np.arange(N_channels), fontsize=15)
    # set the range of the colorbar
    # plt.clim(0, 0.05)
    # Set the colorbar and change the font size
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=25)
    # Set the title
    if title is None:
        plt.title(f"Map for {data_type}", fontsize=20)
    else:
        plt.title(title, fontsize=20)
    plt.show()


def plot_psd_range(data, freqband):
    # Ensure the data is a 3D matrix (trial * phase * freqbin)
    assert data.ndim == 3

    trials, phases, freq_bins = data.shape

    # Create subplots for line plot and violin/histogram plot
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 12),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    violin_parts_list = []
    plotcolors = [
        "blue",
        "orange",
        "red",
        "purple",
        "green",
        "cyan",
        "navy",
        "royalblue",
        "teal",
        "lime",
        "olive",
        "gold",
        "orange",
        "coral",
        "salmon",
        "indianred",
    ]
    xtick = np.linspace(
        band_to_value(freqband)[0], band_to_value(freqband)[1], freq_bins
    )

    # Iterate over the phases
    for p in range(phases):
        phase_data = data[:, p, :]  # Extract data for the current phase

        # Calculate the median and other statistics for line plot
        trial_averaged_median = np.median(phase_data, axis=0)

        lower_ci, upper_ci = bootstrap(phase_data.T, 5000)

        # Plot the median of each trial in each phase
        ax1.plot(xtick, trial_averaged_median, color=plotcolors[p])
        ax1.fill_between(
            xtick, upper_ci, lower_ci, alpha=0.2, color=plotcolors[p]
        )

        # Calculate the median and quartiles for violin plot
        freq_averaged_median = np.median(phase_data, axis=1)

        # Violin plot
        violin_parts = ax2.violinplot(
            freq_averaged_median,
            positions=[p],
            showextrema=False,
            showmedians=False,
        )

        # Change the color of violin plots and create a half violin plot
        for vp in violin_parts["bodies"]:
            violin_parts_list.append(vp)
            vp.set_facecolor(plotcolors[p])
            vp.set_alpha(0.5)

            # Make it a half violin plot
            m = np.mean(vp.get_paths()[0].vertices[:, 0])
            vp.get_paths()[0].vertices[:, 0] = np.clip(
                vp.get_paths()[0].vertices[:, 0], m, np.inf
            )

            # Find the interquartile range within the violin plot's path
            quartile1 = np.percentile(freq_averaged_median, 25)
            quartile3 = np.percentile(freq_averaged_median, 75)
            path = vp.get_paths()[0].vertices
            # Get the indices of the path that correspond to the quartile range
            quartile_mask = (path[:, 1] >= quartile1) & (
                path[:, 1] <= quartile3
            )
            # Color the interquartile range area
            ax2.fill_betweenx(
                path[quartile_mask, 1],
                m,
                path[quartile_mask, 0],
                color="black",
                alpha=0.4,
            )

        # Plot median dot
        ax2.plot(
            p,
            np.median(freq_averaged_median),
            color="white",
            marker="o",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # Histogram
        hist_data, bin_edges = np.histogram(freq_averaged_median, bins=20)
        hist_data = hist_data / hist_data.max() * 0.6
        # Calculate the width of each bin
        bin_width = (
            bin_edges[1] - bin_edges[0]
        ) * 0.4  # Adjust the multiplier to control the bin thickness

        # Draw the bars with spacing
        for j in range(len(hist_data)):
            ax2.barh(
                bin_edges[:-1][j],
                -hist_data[j],
                left=p,
                height=bin_width,
                align="edge",
                color=plotcolors[p],
                edgecolor=plotcolors[p],
            )

    ax1.legend()
    ax1.set_xlabel("Frequency Bin", fontsize=15)
    ax1.set_ylabel("PSD(dB)", fontsize=15)
    ax2.set_xlabel("Phase", fontsize=15)
    ax2.set_ylabel("PSD (dB)", fontsize=15)
    ax2.set_xticks(range(phases))

    return fig, violin_parts_list


def plot_hand_position(accel_data, trial_info, output_dir, targetLocations):
    """
    Plot the hand trajectory based on the accelerometer data. The averaged final location of the hand is also plotted.
    :param accel_data: accelerometer data
    :param trial_info: trial information: end_time, trial_success, condition
    :return:
    """

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Set the colormap
    colors = ["r", "g", "b", "k", "c", "m", "y", "#A2142F"]
    unique_conditions = np.unique(trial_info[:, -1])
    dt = 1 / 2000

    for i in range(len(trial_info) - 1):
        if trial_info[i, -2] == 0:
            continue
        # Get the start and end time of the trial
        end_time = trial_info[i + 1, 0]
        start_time = trial_info[i, -3]
        condition = trial_info[i, -1] - 1

        # Get the accelerometer data for the trial
        handTrajectory = accel_data[start_time:end_time, :]
        handTrajectory = calibrate_accelerometer(handTrajectory)

        # handTrajectory[handTrajectory > 300] = 0

        # Plot the hand trajectory
        handVelocities = np.cumsum(handTrajectory, axis=0) * dt
        handPositions = np.cumsum(handVelocities, axis=0) * dt
        ax.plot(
            handPositions[:, 0],
            handPositions[:, 1],
            color=colors[condition],
            alpha=0.5,
        )
        ax.scatter(
            handPositions[-1, 0],
            handPositions[-1, 1],
            color=colors[condition],
            marker="x",
        )

    # Calculate the average final position of the hand
    finalPositions = targetLocations

    # Plot the average final position of the hand
    for i in range(len(unique_conditions)):
        tep = finalPositions[i]
        ax.scatter(
            tep[0],
            tep[1],
            color=colors[i],
            marker="o",
            s=100,
            label=f"Target {i+1}",
        )

    ax.legend(fontsize=25)
    ax.set_xlabel("X Position (mm)", fontsize=25)
    ax.set_ylabel("Y Position (mm)", fontsize=25)
    ax.set_title("Hand Trajectory", fontsize=30)

    if output_dir is not None:
        plt.savefig(output_dir + "hand_trajectory.png")
    else:
        plt.show()


def calibrate_accelerometer(hand_trajectory):
    """
    Calibrate accelerometer data using PCA to extract the first two principal components
    Parameters:
    - hand_trajectory: numpy array of shape (N, 3) representing the raw accelerometer readings (x, y, z) in mm/s^2
    Returns:
    - calibrated_trajectory: numpy array of shape (N, 3) representing the calibrated accelerometer readings (x, y, z) in mm/s^2
    """
    assert (
        isinstance(hand_trajectory, np.ndarray)
        and hand_trajectory.shape[1] == 3
    ), "Invalid hand_trajectory"

    scaler = StandardScaler()
    rescaledhand_trajectory = scaler.fit_transform(hand_trajectory)
    pca = PCA(n_components=2)
    pca.fit(rescaledhand_trajectory)
    principal_components = pca.transform(hand_trajectory)
    return principal_components


def plot_hand_accelerometer(accel_data, trial_info, output_dir):
    """
    Plot the accelerometer data for each trial
    :param accel_data:
    :param trial_info:
    :param output_dir:
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # Set the colormap
    x_colors = [
        "red",
        "green",
        "blue",
        "chartreuse",
        "coral",
        "cyan",
        "darkblue",
        "darkgreen",
    ]
    y_colors = [
        "darkorange",
        "darkred",
        "darkviolet",
        "deeppink",
        "dodgerblue",
        "firebrick",
        "forestgreen",
        "fuchsia",
    ]

    for i in range(len(trial_info) - 1):
        if trial_info[i, -2] == 0:
            continue
        # Get the start and end time of the trial
        end_time = trial_info[i + 1, 0]
        start_time = trial_info[i, -3]
        condition = trial_info[i, -1] - 1

        trial_data = accel_data[start_time:end_time, :]
        trial_data[trial_data > 300] = 0

        times = np.arange(0, len(trial_data) / 2000, 1 / 2000)

        ax.plot(
            times,
            trial_data[:, 0],
            color=x_colors[condition],
            alpha=0.5,
            label=f"x{condition}",
        )
        ax.plot(
            times,
            trial_data[:, 1],
            color=y_colors[condition],
            alpha=0.5,
            label=f"y{condition}",
        )

    ax.set_xlabel("Time (s)", fontsize=25)
    ax.set_ylabel("Acceleration (mm/s^2)", fontsize=25)
    ax.set_title("Accelerometer Data", fontsize=30)

    if output_dir is not None:
        plt.savefig(output_dir + "accelerometer_data.png")
    else:
        plt.show()


if __name__ == "__main__":
    # Test the hand trajectory plot
    # Generate the random data
    accel_data = np.random.rand(200000, 3)
    # Generate trial_info with 4 colums: end_time, trial_success, condition
    times = np.arange(10000, 200000, 4000).astype(int)
    trial_info = np.zeros((len(times), 3)).astype(int)
    trial_info[:, 0] = times
    trial_info[:, 1] = np.random.randint(1, 2, len(times))
    trial_info[:, 2] = np.random.randint(1, 3, len(times))
    targetLocations = np.random.rand(8, 2)
    output_dir = None
    plot_hand_position(accel_data, trial_info, output_dir, targetLocations)
