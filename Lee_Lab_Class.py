# Import from global libraries
import copy
import os
import pickle
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
from time import time

import numpy as np
import pandas as pd
from dPCA import dPCA
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo,
)
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.svm import SVC
from tqdm import tqdm

from Data_class import Data_Class
from plot_data import (
    add_sig_star,
    plot_beta_hist,
    plot_beta_hist_gng,
    plot_hand_accelerometer,
    plot_hand_position,
    plot_psd_range,
    plot_spectrogram_all,
    plot_violin_hist_half,
)
from PSD_Cal import PSD_cal, Spec_trial

# Import from local libraries
from rm_artifacts import apply_CAR, remove_line, rm_bad_list, rm_bad_list_gng
from Sig_Analysis_Fun import signal_quality_test, simtable
from Time_Warpping_Class import Time_Warping_Task
from utilfun import (
    band_to_value,
    cluster_permutation_test,
    cluster_permutation_test_multi_freq,
    data_combinination,
    extract_data,
    extract_trial_after_ITI,
    extract_trial_GNG,
    get_parameter_grid,
    import_data,
    import_data2,
    import_data_baseline,
    label_select,
    save_figs,
    standarization,
)

pd.options.mode.chained_assignment = None


class TaskOptions:
    def __init__(
        self,
        patNo,
        cod,
        Fs,
        region,
        brainwave,
        output_dir,
        taskID,
        task_folder=None,
        **kwargs,
    ):
        """

        :param patNo: Patient #
        :param cod: Condition- Left/Right hand
        :param Fs: Sampling Frequency
        :param region: Region of interest
        :param brainwave:
        :param output_dir:
        """
        self.patNo = patNo  # Patient #
        self.cod = cod  # Condition- Left/Right hand
        self.Fs = Fs  # Sampling Frequency
        self.region = region  # Region of interest (e.g. Hippocampus, Amygdala, Both, Other or All)
        self.brainwave = (
            brainwave  # Brainwave (e.g. alpha, beta, gamma, high_gamma)
        )
        self.output_dir = output_dir  # Output directory
        self.taskID = taskID  # Task ID (e.g. DelayedReach)
        self.task_folder = task_folder  # Task folder
        self.phase_time_select_method = kwargs.get(
            "phase_time_select_method"
        )  # Phase time select method
        self.referenceMethod = kwargs.get("referenceMethod")  # Reference method

    # Add set function below:


class DelayedReach:
    """
    Delayed Reach class to compute:
    1. PSD of each trial
    2. Perform PSD comparison (including certain plots)
    """

    def __init__(self, options: TaskOptions):
        # Add description of each variable below
        self.movement_type = None
        self.bad_contact = None
        self.dpca = None
        self.dPCA_Z = None
        self.dPCA_Z_shuffle = None
        self.dpca_data = None
        self.re_organized_psd_range = None
        self.spectrogram = {}
        self.dur = None  # Duration of each trial
        self.freq_outlier = None  # Frequency outlier
        self.time_outlier = None  # Time outlier
        self.TimeModel = None  # Time model
        self.psddata_range = None  # PSD data range
        self.quality_test = None  # Signal quality test
        self.new_ele = None  # New electrode
        self.subdata = None  # Sub data
        self.task = None  # Task (e.g. Phase timestamp)
        self.movement = None  # Movement
        self.label = None  # Label
        self.taskID = options.taskID  # Task ID (e.g. DelayedReach)
        self.patNo = options.patNo  # Patient #
        self.cod = options.cod  # Condition (e.g. Left/Right hand)
        self.Fs = options.Fs  # Sampling frequency (e.g. 2000 Hz)
        self.region = options.region  # Region of interest (e.g. "Hippocampus")
        self.brainwave = (
            options.brainwave
        )  # Brainwave band of interest (e.g. beta)
        self.output_dir = options.output_dir  # Output directory
        self.taskfile = options.task_folder  # Task folder
        self.phase_time_select_method = (
            options.phase_time_select_method
        )  # Phase time select method (e.g. 'Same
        # Length')
        self.reference_method = (
            options.referenceMethod
        )  # Reference method (e.g. 'CAR')
        self.screen_size = [476.64, 268.11]  # Screen size
        self.target_location = None  # Target location
        print("Delayed Reach class initialized!")

    def get_target(self):
        """
        Get the target for each trial
        :return:
        """
        self.target_location = defaultdict(list)
        locations = [
            [0, 0.23],
            [0.1626, 0.1626],
            [0.23, 0],
            [0.1626, -0.1626],
            [0, -0.23],
            [-0.1626, -0.1626],
            [-0.23, 0],
            [-0.1626, 0.1626],
        ]
        for i in range(np.unique(self.task[:, -1]).shape[0]):
            self.target_location[i] = self.screen_size * np.array(locations[i])

    def get_data(self):
        self.label = label_select(self.taskID)  # Phase label
        # self.task, data, self.movement, elefile = import_data(self.patNo, self.taskID, self.cod)
        if self.taskfile is None:
            self.task, data, self.movement, elefile = import_data(
                self.patNo, self.taskID, self.cod
            )
        else:
            self.task, data, self.movement, elefile = import_data2(
                self.taskfile
            )
        self.task = self.task[
            :-2, :
        ]  # Some tasks the time will not cover the entire session
        electrode = pd.read_csv(elefile, sep=",", header=0)
        self.subdata, self.new_ele = extract_data(data, self.region, electrode)
        (
            self.subdata,
            self.new_ele,
        ) = self.remove_corrupted_columns_and_update_df(
            self.subdata, self.new_ele
        )
        self.N_channels = self.subdata.shape[1]
        print("Data shape: ", self.subdata.shape)
        print("Session duration: ", self.subdata.shape[0] / self.Fs / 60, "min")
        print("New electrode shape: ", self.new_ele.shape)
        print("Task shape: ", self.task.shape)
        # Calculate the average duration of trials
        avg_dur, max_dur, min_dur = self.calculate_durations()
        print("Average duration of trials: ", avg_dur, "s")
        print("Max duration of trials: ", max_dur, "s")
        print("Min duration of trials: ", min_dur, "s")
        # if self.movement is not empty list
        if self.movement != []:
            print("Movement shape: ", self.movement.shape)
            assert (
                self.movement.shape[0] == self.subdata.shape[0]
            ), "Movement and data shape not match!"
        self.get_target()  # Get the target for each trial
        print("Get data successfully!")

    def calculate_durations(self):
        durations = self.task[1:, 0] - self.task[:-1, 0]
        avg_dur = np.mean(durations) / self.Fs
        max_dur = np.max(durations) / self.Fs
        min_dur = np.min(durations) / self.Fs
        return avg_dur, max_dur, min_dur

    def data_quality_check(self):
        """
        Check the quality of the data
        :return:
        """
        output_dir = self.output_dir + "//Data_Quality_Check//"

        # Create a 2x3 grid of subplots
        self.modify_success_based_on_phase_duration()

        # Second plot is the electrode description
        self.check_signal()

        fig, ax = plt.subplots(1, 1)

        print("Start to plot the first subplot")

        # First plot is statistical text description of the data
        ax.axis("off")
        ax.title.set_text("Statistical Description")
        # Add the statistical description of the data to the first subplot
        # 1. Electrode Shape
        ax.text(0, 0.9, "Recording Areas: " + str(self.new_ele.shape[0]))
        ax.text(0, 0.8, "Total Number of Channels: " + str(self.N_channels))
        # 2. Task Description
        # Calculate the average duration of trials
        avg_dur, max_dur, min_dur = self.calculate_durations()
        # Set numbers to 2 decimal places
        avg_dur = round(avg_dur, 2)
        max_dur = round(max_dur, 2)
        min_dur = round(min_dur, 2)
        ax.text(0, 0.7, "Average duration of trials: " + str(avg_dur) + "s")
        ax.text(0, 0.6, "Max duration of trials: " + str(max_dur) + "s")
        ax.text(0, 0.5, "Min duration of trials: " + str(min_dur) + "s")
        ax.text(0, 0.4, "Total number of trials: " + str(len(self.task[:, -2])))
        ax.text(
            0,
            0.3,
            "Success rate: "
            + str(np.sum(self.task[:, -2]) / len(self.task[:, -2])),
        )

        # Save the figure
        figname = "Statistical Description.png"

        save_figs(output_dir, figname, plt)

        print("Start to plot the second subplot")

        fig, ax = plt.subplots(3, 1, figsize=(15, 10))

        autocorr_data = self.quality_test[0]
        snr_data = self.quality_test[1]

        # Calculate RMS for the first 1 minute of the data
        rms_data = np.sqrt(
            np.mean(self.subdata[: 60 * self.Fs, :] ** 2, axis=0)
        )

        # Subplot - RMS, SNR, and Auto-correlation

        ax[0].plot(snr_data, color="b", label="SNR")
        ax[0].set_ylabel("SNR", fontsize=25)
        ax[0].legend(loc="upper right", fontsize=25)

        ax[1].plot(rms_data, color="g", label="RMS")
        ax[1].set_ylabel("RMS", fontsize=25)
        ax[1].legend(loc="upper right", fontsize=25)
        ax[1].set_xlabel("Channel", fontsize=25)

        for i in range(len(autocorr_data)):
            # Plot the auto-correlation
            ax[2].plot(autocorr_data[i, :], color="gray", alpha=0.2)
        # Plot the mean auto-correlation
        correlation_mean = np.mean(autocorr_data, axis=0)
        ax[2].plot(correlation_mean, color="r", label="Mean Auto-correlation")
        ax[2].set_ylabel("Auto-correlation", fontsize=25)
        ax[2].set_xlabel("Sample", fontsize=25)
        ax[2].set_ylim([-0.3, 0.5])
        ax[2].legend(loc="upper right", fontsize=25)

        # Save the figure
        figname = "RMS, SNR, and Auto-correlation.png"

        plt.tight_layout()

        save_figs(output_dir, figname, plt)

        # Third plot is the raw data visualization for timestamp validation for first 10 trials
        print("Start to plot the third subplot")
        timeTrials = self.task[6:17, 0]

        raw_data = self.subdata[: timeTrials[-1], 0]
        if self.cod == "Left":
            hand_data = self.movement[:, :3]
        else:
            hand_data = self.movement[:, 3:]

        hand_data_to_plot = hand_data[: timeTrials[-1], 0]

        fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        ax[0].plot(raw_data, color="b", label="Raw Data")
        ax[0].set_ylabel("Raw Data", fontsize=25)

        ax[1].plot(hand_data_to_plot, color="r", label="Hand Data")
        ax[1].set_ylabel("Hand Data", fontsize=25)
        ax[1].set_xlabel("Samples", fontsize=25)

        # Shaded the area used for analysis
        for i, timeTrial in enumerate(timeTrials):
            label = "Analysis Area" if i == 0 else None
            ax[0].fill_between(
                [timeTrial - 2 * self.Fs, timeTrial],
                [np.min(raw_data), np.min(raw_data)],
                [np.max(raw_data), np.max(raw_data)],
                color="gray",
                alpha=0.3,
                label=label,
            )
            ax[1].fill_between(
                [timeTrial - 2 * self.Fs, timeTrial],
                [np.min(hand_data_to_plot), np.min(hand_data_to_plot)],
                [np.max(hand_data_to_plot), np.max(hand_data_to_plot)],
                color="gray",
                alpha=0.3,
                label=label,
            )

        ax[0].legend(loc="upper right", fontsize=25)
        ax[1].legend(loc="upper right", fontsize=25)

        # Save the figure
        figname = "Raw Data and Hand Data.png"
        save_figs(output_dir, figname, plt)

        # Fourth plot is the hand trajectory visualization for x and y coordinates
        print("Start to plot the fourth subplot")
        plot_hand_position(
            hand_data, self.task, output_dir, self.target_location
        )

        # Sixth plot is the accelerometer visualization
        print("Start to plot the sixth subplot")
        plot_hand_accelerometer(hand_data, self.task, output_dir)

        # Fifth plot is the cross talk visualization
        print("Start to plot the fifth subplot")
        simtable(
            self.subdata,
            self.subdata,
            "cs",
            output_dir,
            electrode_locations=self.new_ele,
        )

    def remove_corrupted_columns_and_update_df(self, data, df):
        is_corrupted = np.logical_or(
            np.all(data == 0, axis=0), np.any(np.isnan(data), axis=0)
        )
        cleaned_data = data[:, ~is_corrupted]

        corrupted_indices = np.where(is_corrupted)[0]
        for idx in reversed(corrupted_indices):
            df["Channel"] = df["Channel"].apply(
                lambda x: [i for i in x if i != idx + 1]
            )
            df["Channel"] = df["Channel"].apply(
                lambda x: [i - 1 if i > idx + 1 else i for i in x]
            )

        return cleaned_data, df

    def modify_success_based_on_phase_duration(self):
        """
        Modify the success based on the response phase duration,
        if the response phase duration is less than 0.5 second larger than 5 seconds,
        then the trial is considered as unsuccessful
        :return:
        """
        # Get the index of the response phase in the label list
        response_index = self.label.index("Response")
        timestamp = self.task[:, :-2]
        # Get the index of the trials that have response phase duration less than 0.5 second
        bad_trials = np.where(
            (timestamp[1:, 0] - timestamp[:-1, response_index]) < 0.5 * self.Fs
        )[0]
        # Get the index of the trials that have response phase duration larger than 3 seconds
        bad_trials2 = np.where(
            (timestamp[1:, 0] - timestamp[:-1, response_index]) > 2.5 * self.Fs
        )[0]
        bad_trials = np.union1d(bad_trials, bad_trials2)
        if len(bad_trials) > 0:
            self.task[bad_trials, -2] = 0

    def re_reference(self):
        for i in range(0, len(self.task) - 1):  # Define each trial
            sttime = int(self.task[i, 0])
            endtime = int(self.task[i + 1, 0])
            totdata = self.subdata[sttime:endtime, :]
            # Apply Electrode shaft re-reference method
            # if there is re-reference method
            if self.reference_method == "CAR":
                totdata = apply_CAR(
                    totdata, self.new_ele["Channel"], method="mean"
                )
            elif self.reference_method == "CMR":
                totdata = apply_CAR(
                    totdata, self.new_ele["Channel"], method="median"
                )
            elif self.reference_method == "weighted_CAR":
                totdata = apply_CAR(
                    totdata, self.new_ele["Channel"], method="weighted_mean"
                )

            self.subdata[sttime:endtime, :] = totdata

    def remove_line_each_electrode(self, data, ele):
        """
        Remove line noise from each electrode
        :param data: Data
        :param ele: Electrode
        :return: Data after removing line noise
        """
        for e in tqdm(ele["Channel"]):
            data[:, e] = remove_line(data[:, e], [60, 100])
        return data

    def cal_and_remove_bad_rms(self):
        """
        Calculate the RMS for the first 20 seconds of each contact,
        Output the contact is RMS is outside the 1.5*IQR
        :return: bad contact index
        """
        bad_contact = []
        RMS_list = np.sqrt(
            np.mean(self.subdata[: 20 * self.Fs, :] ** 2, axis=0)
        )
        Q1 = np.percentile(RMS_list, 25)
        Q3 = np.percentile(RMS_list, 75)
        IQR = Q3 - Q1
        for i in range(len(RMS_list)):
            if RMS_list[i] > Q3 + 1.5 * IQR or RMS_list[i] < Q1 - 1.5 * IQR:
                bad_contact.append(i)
        return bad_contact

    def pre_process(self):
        start_time = time()
        self.modify_success_based_on_phase_duration()
        # Print the success rate
        print(
            "Success rate: ", np.sum(self.task[:, -2]) / len(self.task[:, -2])
        )
        # Throw error if less than 50% of the trials are successful
        if np.sum(self.task[:, -2]) < 0.5 * len(self.task[:, -2]):
            raise Exception("Less than 50% of the trials are successful")
        self.subdata = self.remove_line_each_electrode(
            self.subdata, self.new_ele
        )
        self.re_reference()
        self.bad_contact = self.cal_and_remove_bad_rms()
        print("Pre-processing finished, time cost: ", time() - start_time, "s")

    def check_signal(self):
        self.quality_test = signal_quality_test(self.subdata, self.output_dir)

    def plot_x_correlation(self, method="cs"):
        simtable(
            self.subdata,
            self.subdata,
            method,
            self.output_dir,
            electrode_locations=self.new_ele,
        )
        print("X-correlation calculated")

    def get_duration(self):
        tasktime = self.task[:, :-2]
        labels = self.label
        self.dur = {}

        for i in range(len(labels)):
            label = labels[i]
            if i != len(labels) - 1:
                durmat = tasktime[:-1, i + 1] - tasktime[:-1, i]
                # If label has keyword Cue
                if "Cue" in label:
                    self.dur[label] = np.min(durmat)
                else:
                    self.dur[label] = np.min(durmat[durmat > (0.5 * self.Fs)])
            else:
                durmat = tasktime[1:, 0] - tasktime[:-1, i]
                self.dur[label] = np.min(durmat[durmat > (1 * self.Fs)])

    def cal_psd_range(self):
        start_time = time()
        self.psddata_range = PSD_cal(
            self.subdata,
            self.movement,
            self.task,
            self.label,
            self.taskID,
            self.cod,
            detrend_lg=False,
            brainwave=self.brainwave,
            psdmethod="multitaper",
            debugst=1,
            dur=self.dur,
            phase_time_select=self.phase_time_select_method,
        )
        print("PSD range calculated, time cost: ", time() - start_time, "s")

    def cal_spectrogram(self, method="multitaper", phase="All"):
        # Assuming Spec_trial is defined elsewhere and accessible
        if phase not in self.label and phase != "All":
            raise ValueError(f"Phase '{phase}' is not recognized.")

        if phase == "All":
            phase_key = phase  # Use "All" as a special key
        else:
            phase_index = self.label.index(phase)
            phase_key = phase  # Could alternatively use phase_index for unique identification

        if phase_key not in self.spectrogram:
            # Assuming phase_index is handled correctly within Spec_trial for "All"
            self.spectrogram[phase_key] = Spec_trial(
                self.subdata,
                self.task,
                self.movement,
                self.cod,
                phaseind=(None if phase == "All" else phase_index),
                method=method,
                label=self.label,
            )

        print(f"Spectrogram calculated for phase {phase} using {method}")
        return self.spectrogram[phase_key]

    def plot_spectrogram(self, phase="All", location="All"):
        # Plot the spectrogram
        if location == "All":
            channels = self.new_ele["Channel"]
        else:
            ele_ind = np.where(self.new_ele["Location"] == location)[0]
            channels = np.concatenate(
                [self.new_ele["Channel"][i] for i in ele_ind]
            )
        plot_spectrogram_all(
            self.spectrogram[phase],
            self.Fs,
            self.task,
            self.movement,
            self.label,
            self.taskID,
            self.output_dir,
            phase_name=phase,
            channels=channels,
        )

    def _get_trial_spectrogram(
        self, band="All", region="All", phase="Response"
    ):
        def _get_spectrogram_all(spectrogram, band="All"):
            if band == "All":
                spectrogram_out = np.concatenate(
                    [
                        spectrogram.alphaspec,
                        spectrogram.betaspec,
                        spectrogram.gammaspec,
                        spectrogram.highgammaspec,
                    ],
                    axis=1,
                )
                # Normalize the spectrogram
                spectrogram_out = (
                    spectrogram_out
                    - np.mean(spectrogram_out, axis=2, keepdims=True)
                ) / np.std(spectrogram_out, axis=2, keepdims=True)
            else:
                spectrogram_out = spectrogram.getband(band, "spec")
                # Normalize the spectrogram
                spectrogram_out = (
                    spectrogram_out
                    - np.mean(spectrogram_out, axis=2, keepdims=True)
                ) / np.std(spectrogram_out, axis=2, keepdims=True)
            if region == "All":
                # Keep only the "good" indices
                spectrogram_out = np.delete(
                    spectrogram_out, self.bad_contact, axis=0
                )

                return spectrogram_out
            else:
                ele_ind = np.where(self.new_ele["Label"] == region)[0]
                channel_ind = np.concatenate(
                    [self.new_ele["Channel"][i] for i in ele_ind]
                )
                # Keep only the "good" indices
                channel_ind = np.setdiff1d(channel_ind, self.bad_contact)

                return spectrogram_out[channel_ind]

        spectrogram_data = self.spectrogram[phase]
        unique_conditions = np.unique(self.task[:, -1])
        spectrogram_out = [0] * len(unique_conditions)

        # For each unique condition, calculate the average spectrogram
        for condition in unique_conditions:
            cond_ind = np.where(self.task[:-1, -1] == condition)[0]
            # Get the spectrogram for the current condition
            spectrogram_out[condition - 1] = [
                _get_spectrogram_all(spectrogram_data[i], band=band)
                for i in cond_ind
            ]

        # Return the spectrogram
        return spectrogram_out

    def _cal_trial_average_spectrogram(self, band="All"):
        spectrogram_data = self._get_trial_spectrogram(band=band)
        # Calculate the average spectrogram for each condition
        spectrogram_avg = [
            np.mean(spectrogram_data[i], axis=0)
            for i in range(len(spectrogram_data))
        ]
        return spectrogram_avg

    def plot_trial_average_spectrogram(self, band="All"):
        spectrogram_avg = self._cal_trial_average_spectrogram(band=band)
        # Plot the average spectrogram
        for movement in np.unique(self.movement_type):
            for channel in range(self.N_channels):
                plot_data = spectrogram_avg[movement - 1][channel]
                vmin = np.percentile(plot_data, 3)
                vmax = np.percentile(plot_data, 97)
                plt.figure()
                im = plt.imshow(plot_data, aspect="auto", origin="lower")
                im.set_clim(vmin, vmax)
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz)")
                plt.title(
                    "Average spectrogram for movement "
                    + str(movement)
                    + " channel "
                    + str(channel)
                )
                figname = "channel " + str(channel) + ".png"
                location = (
                    self.output_dir + self.movement_list[movement - 1] + "\\"
                )
                save_figs(location, figname, plt)
                plt.close()

    def time_warp(self, method):
        """
        Shift the timestamp of the task to align the movement
        :param method: Choose from "ShiftWarp" or "PiecewiseWarp"
        :return:
        """
        self.TimeModel = Time_Warping_Task(
            self.movement, self.task, self.cod, method
        )
        self.TimeModel.timewrap_transform()
        print("Time warping finished")

    def get_outlier(self):
        start_time = time()
        self.time_outlier, self.freq_outlier = rm_bad_list(
            self.psddata_range,
            self.label,
            self.new_ele,
            self.task,
            self.brainwave,
        )
        # Mark all electrodes in the bad contact list as outliers
        for bad_contact in self.bad_contact:
            self.time_outlier[:, bad_contact, :] = True
        print("Outlier calculated, time cost: ", time() - start_time, "s")

    def get_label_from_channel(self, channel_number):
        """
        Get the label from the channel number
        :param channel_number:
        :return:
        """
        # Iterate through each row in the DataFrame
        for index, row in self.new_ele.iterrows():
            # Check if the channel_number is in the "Channel" column of the current row
            if channel_number in row["Channel"]:
                # Return the corresponding "Label" value
                return row["Label"]
        # Return None if the channel_number was not found
        return None

    def get_label_combinations(self, labels):
        """
        Get all combinations of labels
        :return:
        """
        # Find the indices of the elements "Fixation", "Delay", and "Response"
        specific_indices = [
            index
            for index, label in enumerate(labels)
            if label
            in ["Fixation", "Delay", "DelayNG", "Response", "ResponseNG"]
        ]

        # Get all 2-element combinations of the specific indices
        all_combinations = [
            tuple(sorted(comb)) for comb in combinations(specific_indices, 2)
        ]

        # Remove duplicates by converting to a set and then back to a list
        unique_combinations = list(set(all_combinations))

        # Convert to a NumPy array
        combinations_array = np.array(unique_combinations)

        # Sort the array by the first column
        combinations_array = combinations_array[
            combinations_array[:, 0].argsort()
        ]

        return combinations_array

    def plot_psd_range_single(self, channel, freqband, show=False):
        """
        Plot the PSD range for a single channel and a single frequency band
        :param channel:
        :param freqband:
        :return:
        """

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

        start_time = time()
        data_to_plot = self.re_organized_psd_range[freqband][channel]
        channel_name = self.get_label_from_channel(channel)
        labels = self.label
        N_freq_bins = data_to_plot.shape[2]
        freqband_range = band_to_value(freqband)
        xticks = np.linspace(freqband_range[0], freqband_range[1], N_freq_bins)
        groups = self.get_label_combinations(self.label)
        time_outlier_ind = self.time_outlier[0, channel, :]
        freq_outlier_ind = self.freq_outlier[freqband][0, channel, :]
        # Get the outlier indices for the current group
        outliers = np.logical_or(time_outlier_ind, freq_outlier_ind)
        # if too few trials are left after removing outliers, print warning and skip
        if np.sum(~outliers) < 5:
            print(
                "Too few trials left for channel "
                + str(channel)
                + freqband
                + "after removing outliers, skipping..."
            )
            return

        data_to_plot = data_to_plot[~outliers, :]

        plot_max = np.max(data_to_plot) + 0.2
        plot_min = np.min(data_to_plot) - 0.2

        # Calculate p-values for each group
        p_values = []
        for group in groups:
            if group[0] == 1:
                alternative = "greater"
            else:
                alternative = "two-sided"
            group1 = data_to_plot[:, group[0], :]
            group2 = data_to_plot[:, group[1], :]
            p_value = cluster_permutation_test_multi_freq(
                group1.T, group2.T, n_iter=500, alternative=alternative
            )
            p_values.append(p_value)

        # Calculate p-values only for Fixation vs. other phases each frequency bin
        p_values_fixation = []
        # select the groups with Fixation as first column
        fixation_groups = groups[groups[:, 0] == 1]
        for group in fixation_groups:
            group1 = data_to_plot[:, group[0], :]
            group2 = data_to_plot[:, group[1], :]

            p_value_group = np.zeros(N_freq_bins)
            for freq_bin in range(N_freq_bins):
                p_value = cluster_permutation_test(
                    group1[:, freq_bin],
                    group2[:, freq_bin],
                    n_iter=500,
                    alternative="greater",
                )
                p_value_group[freq_bin] = p_value
            p_values_fixation.append(p_value_group)

        p_values_fixation = np.array(p_values_fixation)

        # Start to plot
        fig, violin_parts_list = plot_psd_range(data_to_plot, freqband)
        fig.suptitle(
            channel_name
            + " channel "
            + str(channel)
            + " "
            + freqband
            + " band PSD Analysis",
            fontsize=20,
        )
        ax1, ax2 = fig.axes
        # Add legend to ax1
        line_handles = ax1.get_lines()
        ax1.legend(line_handles, labels, fontsize=15)
        # Add xticks to ax2
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, fontsize=15)

        # Fill the bottom with y-axis and corresponding x-axis if frequency is significant
        for i in range(len(p_values_fixation)):
            if self.label[fixation_groups[i, 1]] == "Response":
                y_plot = plot_max
            else:
                y_plot = plot_min
            for f in range(N_freq_bins - 1):
                if p_values_fixation[i, f] < 0.05:
                    ax1.fill_between(
                        [xticks[f], xticks[f + 1]],
                        [y_plot],
                        [y_plot - 0.4],
                        color=plotcolors[fixation_groups[i, 1]],
                        alpha=0.8,
                    )

        add_sig_star(groups, p_values, ax2, violin_parts_list)
        ax1.set_ylim(plot_min - 0.4, plot_max)
        figtitle = channel_name + "_" + str(channel) + "_" + ".png"
        figlocation = self.output_dir + freqband + "_range\\"
        save_figs(figlocation, figtitle, plt)
        if show:
            plt.show()
        else:
            plt.close()
        print(
            "PSD comparison in range done, time cost: ",
            time() - start_time,
            "s",
        )

    def plotRMSSingle(self, channel, show=False):
        """
        Plot the RMS for a single channel
        :param channel:
        :param show:
        :return:
        """
        # Get the RMS data for the current channel
        startTime = time()
        RMS_data = self.re_organized_rms[channel]
        # Get the channel name
        channel_name = self.get_label_from_channel(channel)
        groups = self.get_label_combinations(self.label)
        # Get the labels
        labels = self.label
        # Get the number of labels
        N_labels = len(labels)
        outliers = self.time_outlier[0, channel, :]
        # if too few trials are left after removing outliers, print warning and skip
        if np.sum(~outliers) < 5:
            print(
                "Too few trials left for channel "
                + str(channel)
                + "after removing outliers, skipping..."
            )
            return

        RMS_data = RMS_data[~outliers]
        plot_max = np.max(RMS_data) + 0.2
        plot_min = np.min(RMS_data) - 0.2

        # Calculate p-values for each group
        p_values = []
        for group in groups:
            alternative = "two-sided"
            group1 = RMS_data[:, group[0]]
            group2 = RMS_data[:, group[1]]
            p_value = cluster_permutation_test(
                group1.T, group2.T, n_iter=500, alternative=alternative
            )
            p_values.append(p_value)

        p_values = np.array(p_values)

        # Plot the RMS data with half-violin half-histoplot
        # Create subplots for line plot and violin/histogram plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 12))
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

        # Iterate over the phases
        for p in range(len(labels)):
            phase_data = RMS_data[:, p]  # Extract data for the current phase

            # Calculate the median and quartiles for violin plot
            freq_averaged_median = phase_data

            # Violin plot
            violin_parts = ax.violinplot(
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
                ax.fill_betweenx(
                    path[quartile_mask, 1],
                    m,
                    path[quartile_mask, 0],
                    color="black",
                    alpha=0.4,
                )

            # Plot median dot
            ax.plot(
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
                ax.barh(
                    bin_edges[:-1][j],
                    -hist_data[j],
                    left=p,
                    height=bin_width,
                    align="edge",
                    color=plotcolors[p],
                    edgecolor=plotcolors[p],
                )

        add_sig_star(groups, p_values, ax, violin_parts_list)
        # Set the x-axis limits
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=15)
        # Set the x-axis label
        ax.set_xlabel("Phase", fontsize=15)
        # Set the y-axis label
        ax.set_ylabel("RMS", fontsize=15)
        # Set the y-axis limits
        ax.set_ylim(plot_min, plot_max)
        # Set Figure Title
        fig.suptitle(f"RMS for {channel_name} channel {channel}", fontsize=20)

        # Save the figure
        fig_title = f"{channel_name}_{channel}_RMS.png"
        fig_location = self.output_dir + "RMS\\"
        save_figs(fig_location, fig_title, plt)
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()
        print(
            f"RMS plot for channel {channel} done, time cost:"
            f" {time() - startTime} s"
        )

    def plot_psd_range_all(self):
        """
        Plot the PSD range for all channels and all frequency bands
        :return:
        """
        for freqband in self.brainwave:
            for channel in range(self.N_channels):
                if channel in self.bad_contact:
                    continue
                else:
                    self.plot_psd_range_single(channel, freqband)

    def plot_rms_all(self):
        """
        Plot the RMS for all channels
        :return:
        """
        for channel in range(self.N_channels):
            if channel in self.bad_contact:
                continue
            else:
                self.plotRMSSingle(channel)

    def compare_psd_range(self):
        self.banddata = PSD_comparison_range(
            self.psddata_range,
            self.label,
            self.brainwave,
            self.new_ele,
            self.time_outlier,
            self.freq_outlier,
            self.output_dir,
        )

    def reorganize_data(self, data, type="PSD"):
        f"""
        :param data: a list of PSD data objects, with input as PSD or spectrogram
        PSD: data[trial][brainwave][channel * frequency]
        Spectrogram: data[trial][brainwave][channel * frequency * time]
        Reorganize the data into a dictionary with keys as brainwaves and values as a list of numpy arrays
        :return: output: a dictionary with keys as brainwaves and values as a list of numpy arrays:
        output[brainwave][channel][trial * phase * frequency]
        """
        start_time = time()
        brainwaves = self.brainwave
        phases = self.label
        # Get the number of channels from the first data object
        num_channels = (
            self.N_channels
        )  # Assuming 'alpha' attribute is list-like and represents channels

        # Find out how many unique trials and phases there are
        unique_trials = set([d.trial for d in data])

        if type == "PSD":
            # Calculate number of frequency bins from first data object
            frequency_bins = {
                brainwave: len(data[0].getband(brainwave)[0])
                for brainwave in brainwaves
            }

            # Create arrays to hold reorganized data for each bandpass
            output = {
                brainwave: [
                    np.zeros(
                        (
                            len(unique_trials),
                            len(phases),
                            frequency_bins[brainwave],
                        )
                    )
                    for _ in range(num_channels)
                ]
                for brainwave in brainwaves
            }

            # Map phase names to indices
            phase_to_index = {
                phase: index for index, phase in enumerate(phases)
            }

            # Loop over each trial in the data
            for trial in data:
                trial_index = trial.trial  # assuming trials are 0-indexed
                phase_index = phase_to_index[trial.phase]

                # Loop over each channel
                for channel in range(num_channels):
                    for brainwave in brainwaves:
                        # Store the data for each frequency bin in the correct place in the arrays
                        output[brainwave][channel][
                            trial_index, phase_index, :
                        ] = trial.getband(brainwave)[channel]
            print("Data reorganized, time cost: ", time() - start_time, "s")
            self.re_organized_psd_range = output

        elif type == "RMS":
            # Create arrays to hold reorganized data
            output = [
                np.zeros((len(unique_trials), len(phases)))
                for _ in range(num_channels)
            ]
            # Map phase names to indices
            phase_to_index = {
                phase: index for index, phase in enumerate(phases)
            }
            # Loop over each trial in the data
            for trial in data:
                trial_index = trial.trial
                phase_index = phase_to_index[trial.phase]
                # Loop over each channel
                for channel in range(num_channels):
                    output[channel][trial_index, phase_index] = trial.getRMS()[
                        channel
                    ]
            print("Data reorganized, time cost: ", time() - start_time, "s")
            self.re_organized_rms = output

    def plot_fix_vs_delay(self, region, phase1="Fixation", phase2="Delay"):
        print("Plotting phase comparison for region " + region)
        start_time = time()
        data = self.re_organized_psd_range
        # Get fixation index
        fix_index = self.label.index(phase1)
        # Get delay index
        delay_index = self.label.index(phase2)
        # Extract channel number with region
        electrodes = list(
            self.new_ele[
                self.new_ele["Location"].str.contains(region, case=False)
            ]["Channel"]
        )
        electrodes_names = list(
            self.new_ele[
                self.new_ele["Location"].str.contains(region, case=False)
            ]["Label"]
        )
        for brainwave in self.brainwave:
            if phase1 == "Fixation" and brainwave == "beta":
                alternative = "greater"
            else:
                alternative = "two-sided"
            plot_index = 0
            plot, ax1 = plt.subplots(figsize=(10, 7), layout="constrained")
            positions = []
            electrode_positions = []
            electrodes_names_to_plot = []
            channel_index = 0
            for channels in electrodes:
                # Get channel name
                channel_name_tep = electrodes_names[channel_index]
                # Delete the outlier channel index
                channels = np.setdiff1d(channels, self.bad_contact)
                for channel in channels[0:3]:
                    if plot_index > 15:
                        break
                    fix_data = data[brainwave][channel][:, fix_index, :]
                    delay_data = data[brainwave][channel][:, delay_index, :]

                    fix_time_outlier = self.time_outlier[fix_index, channel]
                    fix_freq_outlier = self.freq_outlier[brainwave][
                        fix_index, channel
                    ]

                    outliers = np.logical_or(fix_time_outlier, fix_freq_outlier)

                    # Select the data where outliers are false
                    fix_data = fix_data[~outliers, :]
                    delay_data = delay_data[~outliers, :]

                    p_value = cluster_permutation_test_multi_freq(
                        fix_data.T,
                        delay_data.T,
                        n_iter=5000,
                        alternative=alternative,
                    )

                    fix_data_median = np.median(fix_data, axis=1)
                    delay_data_median = np.median(delay_data, axis=1)

                    position = plot_violin_hist_half(
                        fix_data_median,
                        delay_data_median,
                        p_value,
                        ax1,
                        plot_index * 2,
                        electrode_name=channel_name_tep,
                        labels=[phase1, phase2],
                    )

                    positions.extend(position)
                    # Add 1/2/3 to channel name
                    channel_name = channel_name_tep + str((plot_index) % 3)
                    electrodes_names_to_plot.append(channel_name)
                    electrode_positions.append(np.mean(position))
                    plot_index += 1
                channel_index += 1
            ax1.set_xticks(positions)
            ax1.set_xticklabels([" ", " "] * plot_index, rotation=45)
            ax1.set_title(
                region + " " + brainwave + " " + phase1 + " vs " + phase2,
                fontsize=18,
            )
            ax1.set_ylabel("PSD Power(dB)", fontsize=15)
            ax1.set_xlabel("Phase/Electrodes", loc="right", fontsize=15)
            for name, pos in zip(electrodes_names_to_plot, electrode_positions):
                pos_in_axes_coords = ax1.transLimits.transform((pos, 0))[0]
                ax1.text(
                    pos_in_axes_coords,
                    -0.02,
                    name,
                    ha="center",
                    va="top",
                    fontsize=10,
                    transform=ax1.transAxes,
                )
            plt.savefig(
                self.output_dir
                + region
                + "_"
                + brainwave
                + phase1
                + "_"
                + phase2
                + ".png"
            )
            plt.close()
        print("Plotting done, time cost: ", time() - start_time, "s")

    def plot_channel_psd_histo(self):
        plot_channel = self.new_ele[
            self.new_ele["Location"].str.contains("Hippocampus", case=False)
        ]

        # Plot the histogram of distribution of band power for the first channel in hippocampus channel
        for e in plot_channel["GridId"]:
            plot_beta_hist(
                self.psddata_range,
                self.label,
                plot_channel["Channel"][e][0],
                self.output_dir,
            )

    def save_to_json(self):
        start_time = time()
        # Save self to json file
        # Replicate the object to avoid saving the whole object
        self_copy = copy.deepcopy(self)
        self_copy.psddata_over_time = None
        self_copy.psddata_range = None
        # self_copy.quality_test = None

        with open(self.output_dir + "pat.pkl", "wb") as fp:
            pickle.dump(self_copy, fp)
        print("Object saved, time cost: ", time() - start_time, "s")

    def plot_hist_all_channels_all_freqs(self):
        """
        Plot the histogram of all channels and all frequencies,
        Each figure represents one frequency band
        Each subplot represents one channel
        :return:
        """
        data_to_plot = self.re_organized_psd_range
        brainwaves = self.brainwave
        num_channels = len(data_to_plot["beta"])
        num_rows = int(np.sqrt(num_channels)) + 1
        num_cols = num_rows
        # phase_labels = self.movement_list
        if self.taskID == "Move":
            if self.patNo != 57 and self.patNo != 62:
                phase_labels = np.array(["Left", "Right", "", "", "Nothing"])
            else:
                phase_labels = np.array(["Left", "Right", "Nothing"])
        else:
            phase_labels = np.array(["", "Fixation", "", "Delay", "Response"])
        for brainwave in brainwaves:
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
            for channel in range(num_channels):
                row = channel // num_rows
                col = channel % num_cols
                data_to_plot_channel = data_to_plot[brainwave][channel]
                time_outlier_ind = self.time_outlier[:, channel, :]
                freq_outlier_ind = self.freq_outlier[brainwave][:, channel, :]
                outlier_ind = time_outlier_ind | freq_outlier_ind
                outlier_ind = outlier_ind.T[:, 0]

                data_to_plot_channel = data_to_plot_channel[~outlier_ind, :]
                data_to_plot_single_phase = np.zeros(
                    (data_to_plot_channel.shape[0], len(phase_labels))
                )

                # First row as baseline, calculate the relative change of power
                baseline = data_to_plot_channel[:, 0, :]
                baseline = 10 ** (baseline / 10)
                # Normalize the baseline
                baseline = np.median(baseline, axis=1)
                for p in range(0, len(phase_labels)):
                    if phase_labels[p] != "":
                        tep = np.median(
                            data_to_plot_channel[:, p + 1, :], axis=1
                        )
                        tep = 10 ** (tep / 10)
                        data_to_plot_single_phase[:, p] = (
                            np.log10(tep / baseline) * 10
                        )
                        # Make the same index of element in data_to_plot_single_phase with tep==1 as 0
                        data_to_plot_single_phase[tep == 1, p] = 0
                plot_PSD_histo_comparison(
                    data_to_plot_single_phase,
                    phase_labels,
                    brainwave,
                    axs=axs,
                    row=row,
                    col=col,
                )
                axs[row, col].set_title(str(channel))
                # Plot a vertical line at 1
                axs[row, col].axvline(x=1, color="k", linestyle="--")
            # Adjust the spacing between subplots
            plt.subplots_adjust(
                left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.7
            )
            fig.suptitle(brainwave)
            fig.legend(phase_labels[phase_labels != ""], loc="upper right")
            plt.savefig(
                self.output_dir + brainwave + "_hist_all_channels_all_freqs.png"
            )
            plt.close()

    def clustering(self, data: dict = None):
        """
        Perform clustering on the data, restructure the data to be 2D with rows as trials and columns as features
        :return:
        """
        if data is None:
            data = self.re_organized_psd_range
        self.combined_data = data_combinination(**data)
        # Delete all the rows with all 0
        for i in range(len(self.combined_data)):
            self.combined_data[i] = self.combined_data[i][
                ~np.all(self.combined_data[i] == 0, axis=1), :
            ]
            # Add labels to the last column
            self.combined_data[i] = np.hstack(
                (
                    self.combined_data[i],
                    np.ones((self.combined_data[i].shape[0], 1)) * i,
                )
            )
        # Combine all the data and labels as last column
        self.combined_data = np.vstack(self.combined_data)
        self.backupdata = self.combined_data

        return self.combined_data

    def re_balance_data(self, train_data, train_label):
        ros = RandomOverSampler(random_state=42)
        return ros.fit_resample(train_data, train_label)

    def divide_train_test(self, random_state=42):
        """
        Divide the data into training and testing set
        :return:
        """
        # def feature_selection(data, method):
        #     '''
        #     Remove the feature with low variance
        #     :param data:
        #     :param method:
        #     :return:
        #     '''
        data = self.combined_data
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=random_state
        )
        label = data[:, -1]
        for train_index, test_index in sss.split(data, label):
            train, test = data[train_index], data[test_index]
        # Get the training data and labels
        self.train_data = train[:, :-1]
        self.train_label = train[:, -1]
        # Get the testing data and labels
        self.test_data = test[:, :-1]
        self.test_label = test[:, -1]
        # Re-balance the training data
        self.train_data, self.train_label = self.re_balance_data(
            self.train_data, self.train_label
        )

    def standardize_data(self):
        """
        Standardize the data
        :return:
        """
        scaler = standarization(self.train_data)
        self.train_data = scaler.transform(self.train_data)
        self.test_data = scaler.transform(self.test_data)

    def dimension_reduction(self, method="PCA", n_dimensions=2, verbose=False):
        """

        :param method:
        :param n_dimensions:
        :return:
        """
        if method == "PCA":
            pca = PCA(n_components=n_dimensions)
            pca.fit(self.train_data)
            self.train_data_reduced = pca.transform(self.train_data)
            self.test_data_reduced = pca.transform(self.test_data)
        elif method == "LDA":
            lda = LDA(n_components=n_dimensions)
            lda.fit(self.train_data, self.train_label)
            self.train_data_reduced = lda.transform(self.train_data)
            self.test_data_reduced = lda.transform(self.test_data)
        else:
            if verbose:
                print("Dimension reduction method not supported!")
                print("Use unreduced data instead...")
            self.train_data_reduced = self.train_data
            self.test_data_reduced = self.test_data

    def train_model(self, model="SVM", **kwargs):
        """
        Train the model
        :param model: 'SVM' or 'LDA' or 'QDA'
        :return:
        """
        if model == "SVM":
            parameter_grid = kwargs["parameter_grid"]
            self.TimeModel = GridSearchCV(SVC(), parameter_grid, refit=True)
            self.TimeModel.fit(self.train_data_reduced, self.train_label)
        elif model == "LDA":
            parameter_grid = kwargs["parameter_grid"]
            self.TimeModel = GridSearchCV(LDA(), parameter_grid, refit=True)
            self.TimeModel.fit(self.train_data_reduced, self.train_label)
        elif model == "QDA":
            parameter_grid = kwargs["parameter_grid"]
            self.TimeModel = GridSearchCV(QDA(), parameter_grid, refit=True)
            self.TimeModel.fit(self.train_data_reduced, self.train_label)
        else:
            print("Model not supported!")
            print("Use SVM instead...")
            parameter_grid = get_parameter_grid("SVM")
            self.TimeModel = GridSearchCV(SVC(), parameter_grid, refit=True)
            self.TimeModel.fit(self.train_data_reduced, self.train_label)

    def test_model(self):
        """
        Test the model
        :return:
        """
        predicted_label = self.TimeModel.predict(self.test_data_reduced)
        # print("Best parameters: ", self.TimeModel.best_params_)
        accuracy = accuracy_score(self.test_label, predicted_label)
        return accuracy

    def train_test_dataset(
        self,
        dataset,
        dr_method="None",
        n_dimensions=2,
        model="SVM",
        n_fold=5,
        random_state=42,
    ):
        """
        :param dataset:
        :param dr_method: dimension reduction method
        :param n_dimensions:
        :param model:
        :param n_fold:
        :return:
        """
        test_score = []
        for n in range(n_fold):
            self.combined_data = dataset
            self.divide_train_test(random_state=random_state + n - 1)
            self.standardize_data()

            ml_models = model
            # No dimension reduction
            self.dimension_reduction(
                method=dr_method, n_dimensions=n_dimensions
            )
            param_grid = get_parameter_grid(ml_models)

            self.train_model(model=ml_models, parameter_grid=param_grid)
            test_score.append(self.test_model())

        return np.average(test_score)

    def _construct_dPCA_dataset(
        self,
        datatype="psd",
        task_to_discriminate="Response",
        shuffle=False,
        dur=1.5,
        **kwargs,
    ):
        """
        Construct the dataset for dPCA
        :param datatype:
        :param task_to_discriminate:
        :param shuffle:
        :return:
        """
        if datatype == "raw":
            data = copy.deepcopy(self.subdata)
            # Remove the contacts in self.bad_contact
            data = np.delete(data, self.bad_contact, axis=1)
            task = self.task
            column_ID = self.label.index(task_to_discriminate)
            # Rearrange the task to extract only the response column and the ITI column and the last column
            task = np.array(
                [
                    (
                        int(task[i, column_ID] + 0.5 * self.Fs),
                        int(task[i, column_ID] + (dur + 0.5) * self.Fs),
                        task[i, -1],
                    )
                    for i in range(1, len(task))
                ]
            )

            # Construct the dataset as [trial, channel, label, time]
            phaselabels = np.unique(task[:, -1])
            if shuffle:
                task[:, -1] = np.random.permutation(task[:, -1])
            dataset = [[] for _ in range(len(phaselabels))]
            for i in range(len(phaselabels)):
                for k in range(len(task)):
                    if task[k, -1] == phaselabels[i]:
                        tep_data = data[task[k, 0] : task[k, 1], :].T
                        # # Normalize the data for each row
                        dataset[i].append(tep_data)

        elif datatype == "psd":

            def re_arrange_data(data):
                # Now, reshape the filtered_data array
                reshaped_data = np.reshape(data, (-1, data.shape[-1]))

                return reshaped_data

            band = kwargs.get("band", "All")
            region = kwargs.get("region", "All")
            dataset = self._get_trial_spectrogram(band, region=region)
            for i in range(len(dataset)):
                # Combine the first and second axis of each trial
                dataset[i] = np.array(
                    [
                        re_arrange_data(dataset[i][j])
                        for j in range(len(dataset[i]))
                    ]
                )

        # Save the unchunked dataset for later use
        if not shuffle:
            self.dpca_data = copy.deepcopy(dataset)
            for i in range(len(self.dpca_data)):
                self.dpca_data[i] = np.array(self.dpca_data[i])
                self.dpca_data[i] = np.transpose(self.dpca_data[i], (1, 0, 2))

        # Rearrange the data to [trial, channel, label, time]
        min_length = (
            np.min([len(dataset[i]) for i in range(len(dataset))]) - 3
        )  # Save the last 3 trials for testing
        dataset_training = [
            dataset[i][0:min_length] for i in range(len(dataset))
        ]
        dataset_testing = [dataset[i][min_length:] for i in range(len(dataset))]
        dataset_training = np.array(dataset_training)
        # Reshape the dataset to [trial, channel, label, time]
        dataset_training = np.transpose(dataset_training, (1, 2, 0, 3))

        return dataset_training, dataset_testing

    def dPCA(
        self,
        datatype="psd",
        task_to_discriminate="Response",
        shuffle=False,
        dur=1.5,
        sig_mask=True,
        n_component=10,
        **kwargs,
    ):
        def _plot_dPCA_axis(axis=0, sig_mask=True):
            # plot results
            time = np.arange(T) / self.Fs

            plt.figure(figsize=(16, 7))
            plt.subplot(131)

            for s in range(S):
                plt.plot(time, Z["t"][axis, s])

            plt.title("1st time component", fontsize=20)
            plt.xlabel("Time (ms)", fontsize=15)
            plt.ylabel("dPCA axis", fontsize=15)

            plt.subplot(132)

            for s in range(S):
                plt.plot(time, Z["s"][axis, s])
            if sig_mask:
                plt.imshow(
                    significance_masks["s"][axis][None, :],
                    extent=[0, T, np.amin(Z["s"]) - 1, np.amin(Z["s"]) - 0.5],
                    aspect="auto",
                    cmap="gray_r",
                    vmin=0,
                    vmax=1,
                )
            plt.ylim(
                [
                    np.amin(Z["s"]) - 0.1 * np.amin(Z["s"]),
                    np.amax(Z["s"]) + 0.1 * np.amin(Z["s"]),
                ]
            )

            plt.title("1st stimulus component", fontsize=20)
            plt.xlabel("Time (ms)", fontsize=15)
            plt.subplot(133)

            for s in range(S):
                plt.plot(time, Z["st"][axis, s])

            dZ = np.amax(Z["st"]) - np.amin(Z["st"])
            if sig_mask:
                plt.imshow(
                    significance_masks["st"][axis][None, :],
                    extent=[
                        0,
                        T,
                        np.amin(Z["st"]) - dZ / 10.0,
                        np.amin(Z["st"]) - dZ / 5.0,
                    ],
                    aspect="auto",
                    cmap="gray_r",
                    vmin=0,
                    vmax=1,
                )

            plt.ylim(
                [np.amin(Z["st"]) - dZ / 10.0, np.amax(Z["st"]) + dZ / 10.0]
            )

            plt.title("1st mixing component", fontsize=20)
            if shuffle:
                plt.suptitle("Shuffled", fontsize=20)
            plt.legend(["Right", "Left", "Nothing"], fontsize=15)
            plt.xlabel("Time (ms)", fontsize=15)
            plt.show()

        dPCA_dataset, dPCA_testing = self._construct_dPCA_dataset(
            datatype=datatype,
            task_to_discriminate=task_to_discriminate,
            shuffle=shuffle,
            dur=dur,
            **kwargs,
        )
        if dPCA_dataset.size == 0:
            return None
        samples, N, S, T = dPCA_dataset.shape
        # trial-average data
        dPCA_dataset_R = np.mean(dPCA_dataset, 0)
        # center data
        dPCA_dataset_R -= np.mean(dPCA_dataset_R.reshape((N, -1)), 1)[
            :, None, None
        ]

        self.dpca = dPCA.dPCA(
            labels="st", regularizer="auto", n_components=n_component
        )
        self.dpca.protect = ["t"]

        Z = self.dpca.fit_transform(dPCA_dataset_R, dPCA_dataset)

        if shuffle:
            self.dPCA_Z_shuffle = Z
        else:
            self.dPCA_Z = Z

        if sig_mask:
            significance_masks = self.dpca.significance_analysis(
                dPCA_dataset_R,
                dPCA_dataset,
                n_shuffles=10,
                n_splits=10,
                n_consecutive=10,
            )
        _plot_dPCA_axis(sig_mask=sig_mask)

    def dPCA_transform(
        self,
        datatype="psd",
        task_to_discriminate="Response",
        shuffle=False,
        dur=1.5,
        sig_mask=True,
        n_component=10,
        **kwargs,
    ):
        self.dPCA(
            datatype=datatype,
            task_to_discriminate=task_to_discriminate,
            shuffle=shuffle,
            dur=dur,
            sig_mask=sig_mask,
            n_component=n_component,
            **kwargs,
        )
        if self.dpca_data[0].size == 0:
            return None
        # Combine the test data into a single array
        test_data = np.concatenate(
            (self.dpca_data[0], self.dpca_data[1], self.dpca_data[2]), axis=1
        )
        # Create the label array
        label = np.concatenate(
            (
                np.zeros(self.dpca_data[0].shape[1]),
                np.ones(self.dpca_data[1].shape[1]),
                np.ones(self.dpca_data[2].shape[1]) * 2,
            ),
            axis=0,
        )

        test_data_transformed = self.dpca.transform(test_data)["st"]
        test_data_transformed = test_data_transformed.transpose(1, 2, 0)
        return test_data_transformed, label

    def excute_Delayed_Reach(self, save=False):
        """

        :param save:
        :return:
        """

        print(self.taskID + " Analysis Starts...")
        self.get_data()
        self.pre_process()
        self.check_signal()
        self.plot_x_correlation()
        self.get_duration()
        self.cal_psd_range()
        self.reorganize_data(data=self.psddata_range, type="PSD")
        self.reorganize_data(data=self.psddata_range, type="RMS")
        self.get_outlier()
        # self.plot_psd_range_all()

        # Save self to pickle file for later plot use
        if save:
            self.save_to_json()

        self.plot_fix_vs_delay("All", phase1="Fixation", phase2="Cue1")
        self.plot_fix_vs_delay("All", phase1="Delay", phase2="Cue1")
        self.plot_fix_vs_delay("All", phase1="Fixation", phase2="Delay")

        # self.compare_psd_range()


class Movement(DelayedReach):
    def __init__(self, options: TaskOptions):
        super().__init__(options)
        self.backupdata = None
        self.accuracy = None
        self.train_label = None
        self.test_label = None
        self.predicted_label = None
        self.test_data = None
        self.train_data = None
        self.test_data_reduced = None
        self.combined_data = None
        self.train_data_reduced = None
        self.movement = None
        self.task = None
        self.label = None
        self.taskID = options.taskID  # Task ID (e.g. DelayedReach)
        self.patNo = options.patNo  # Patient #
        self.cod = options.cod  # Condition (e.g. Left/Right hand)
        self.Fs = options.Fs  # Sampling frequency (e.g. 2000 Hz)
        self.region = options.region  # Region of interest (e.g. "Hippocampus")
        self.brainwave = (
            options.brainwave
        )  # Brainwave band of interest (e.g. beta)
        self.output_dir = options.output_dir  # Output directory
        self.taskfile = options.task_folder  # Task folder
        self.phase_time_select_method = (
            options.phase_time_select_method
        )  # Phase time select method (e.g. 'Same
        # Length')
        self.reference_method = (
            options.referenceMethod
        )  # Reference method (e.g. 'CAR')
        self.movement_type = None
        if self.patNo != 57 and self.patNo != 62:
            self.movement_list = [
                "Left",
                "Right",
                "ImagineLeft",
                "ImagineRight",
                "Rest",
            ]
        else:
            self.movement_list = ["Left", "Right", "Rest"]

    def get_target(self):
        """
        Get the target for each trial
        :return:
        """
        self.target_location = defaultdict(list)
        locations = [[0.23, 0], [-0.23, 0], [0, 0]]
        for i in range(np.unique(self.task[:, -1]).shape[0]):
            self.target_location[i] = self.screen_size * np.array(locations[i])

    def modify_success_based_on_phase_duration(self):
        """
        Modify the success based on the response phase duration,
        if the response phase duration is less than 0.5 second larger than 5 seconds,
        then the trial is considered as unsuccessful
        :return:
        """
        # Get the index of the response phase in the label list
        response_index = self.label.index("Response")
        timestamp = self.task[:, :-2]
        # Get the index of the trials that have response phase duration less than 0.5 second
        bad_trials = np.where(
            (timestamp[1:, 0] - timestamp[:-1, response_index]) < 0.5 * self.Fs
        )[0]
        # Get the index of the trials that have response phase duration larger than 5 seconds
        bad_trials2 = np.where(
            (timestamp[1:, 0] - timestamp[:-1, response_index]) > 5 * self.Fs
        )[0]
        bad_trials = np.union1d(bad_trials, bad_trials2)
        if len(bad_trials) > 0:
            self.task[bad_trials, -2] = 0

    def get_outlier(self):
        """
        Get outlier in time and frequency domain
        :return:
        """
        self.movement_type = self.task[:-1, -1]  # Python index starts from 0
        self.time_outlier, self.freq_outlier = rm_bad_list(
            self.psddata_range,
            self.label,
            self.new_ele,
            self.task,
            self.brainwave,
            movement_type=self.movement_type,
        )
        for bad_contact in self.bad_contact:
            self.time_outlier[:, bad_contact, :] = True

    def reorganize_data(self, data):
        """
        Reorganize the data into a dictionary with keys as brainwaves and values as a list of numpy arrays
        :return: output: a dictionary with keys as brainwaves and values as a list of numpy arrays:
        output[brainwave][channel][trial * class* frequency]
        """
        start_time = time()
        brainwaves = self.brainwave
        phases = ["Fixation"] + self.movement_list
        # Get the number of channels from the first data object
        num_channels = len(
            data[0].getband("alpha")
        )  # Assuming 'alpha' attribute is list-like and represents channels

        # Find out how many unique trials and phases there are
        unique_trials = set([d.trial for d in data])

        # Calculate number of frequency bins from first data object
        frequency_bins = {
            brainwave: len(data[0].getband(brainwave)[0])
            for brainwave in brainwaves
        }

        # Create arrays to hold reorganized data for each bandpass
        output = {
            brainwave: [
                np.zeros(
                    (len(unique_trials), len(phases), frequency_bins[brainwave])
                )
                for _ in range(num_channels)
            ]
            for brainwave in brainwaves
        }

        # Loop over each trial in the data
        for trial in data:
            trial_index = trial.trial  # assuming trials are 1-indexed
            phase_label = trial.phase
            if phase_label == "Response":
                phase_index = trial.seqid
            elif phase_label == "Fixation":
                phase_index = 0
            else:
                continue

            # Loop over each channel
            for channel in range(num_channels):
                for brainwave in brainwaves:
                    # Store the data for each frequency bin in the correct place in the arrays
                    output[brainwave][channel][
                        trial_index, phase_index, :
                    ] = trial.getband(brainwave)[channel]
        print("Data reorganized, time cost: ", time() - start_time, "s")
        return output

    def excute_Movement(self, save=False):
        print("Movement Analysis Starts...")
        self.get_data()
        self.pre_process()
        print("Preprocess Done!\nStart Signal Checking...")
        self.check_signal()
        print(
            "Signal Checking done!\nStart PSD calculation and outlier"
            " removal..."
        )
        self.plot_x_correlation()
        self.get_duration()
        self.cal_psd_range()
        self.get_outlier()

        self.re_organized_psd_range = self.re_organized_psd_range()

        if save:
            self.save_to_json()


class Baseline(DelayedReach):
    def __init__(self, options: TaskOptions):
        super().__init__(options)
        self.baseline = None
        self.label = None
        self.taskID = options.taskID

    def get_data(self):
        self.label = label_select(self.taskID)  # Phase label
        # self.task, data, self.movement, elefile = import_data(self.patNo, self.taskID, self.cod)
        data, elefile = import_data_baseline(self.patNo, self.taskID, self.cod)
        electrode = pd.read_csv(elefile, sep=",", header=0)
        self.subdata, self.new_ele = extract_data(data, self.region, electrode)
        (
            self.subdata,
            self.new_ele,
        ) = self.remove_corrupted_columns_and_update_df(
            self.subdata, self.new_ele
        )
        print("Data shape: ", self.subdata.shape)
        print("Session duration: ", self.subdata.shape[0] / self.Fs / 60, "min")
        print("New electrode shape: ", self.new_ele.shape)
        # Create artificial task variable, divide the data into 5s trials
        self.task = np.array(
            [
                np.arange(0, self.subdata.shape[0] - 5 * self.Fs, 5 * self.Fs),
                np.arange(5 * self.Fs, self.subdata.shape[0], 5 * self.Fs),
            ]
        ).T

        print("Get data successfully!")

    def pre_process(self):
        self.subdata = remove_line(self.subdata, [60, 100, 120])
        print("Pre-processing finished")
