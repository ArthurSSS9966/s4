import copy
import glob
import math
import os
import pickle
import random
import time
from itertools import combinations
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import mat73
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as io
from joblib import Parallel, delayed
from scipy.ndimage import label, measurements
from scipy.signal import butter, filtfilt
from scipy.stats import norm, wilcoxon, zscore
from sklearn.preprocessing import StandardScaler

n_cpus = os.cpu_count() // 2

"""
Place to tune the the threshold for the duration each trial
"""


def save_load_data(data, ele, patNo, taskID, cod):
    """

    :param data:
    :param elefile:
    :param datafilename:
    :param elefilename:
    :return:
    """
    file_loc = "D:\\Blcdata\\output\\"
    patnb = "P" + "0" + str(patNo) + "_"
    patfolder = patnb + taskID
    if cod:
        patfolder = patnb + taskID + "\\" + cod
    file_fold = file_loc + patfolder
    datafilename = os.path.join(file_fold, "subdata.npy")
    elefilename = os.path.join(file_fold, "subele.npy")
    if os.path.exists(datafilename):
        with open(datafilename, "wb") as f:
            subdata = np.load(f)
            new_ele = pd.read_pickle("subdataDR_ele.pkl")
    else:
        with open(datafilename, "wb") as f:
            np.save(f, data)
            subdata = data
            ele.to_pickle(elefilename)
            new_ele = ele
    return subdata, new_ele


def save_figs(location, fig_name, plot):
    if not os.path.exists(location):
        os.makedirs(location)
    plot.savefig(os.path.join(location, fig_name))


def import_data(patNo, taskID, cod):
    ## Specify file location
    print("Start Data Import...Load Raw Data")
    file_loc = "D:\\Blcdata\\output\\"
    patnb = "P" + "0" + str(patNo) + "_"
    patfolder = patnb + taskID
    if cod:
        patfolder = patnb + taskID + "\\" + cod
    file_fold = file_loc + patfolder

    ## Load Data and clear data
    """
    task: nparray with task type (col 7), phase time (col 1-5), success or not (col 6)
    data: nparray preprocessed data with 60Hz removal containing n channels
    electrode: Dataframe with Location, Hemisphere, Label, 
    """
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(
        initialdir=file_fold
    )  # show an "Open" dialog box and return the path to the selected file
    taskname = filename.replace("data.mat", "task.mat")
    movefile = filename.replace("data.mat", "move.mat")
    elecnm = "gridmap.p0" + str(patNo) + ".txt"
    elefile = file_fold + "\\" + elecnm

    move = []

    task = io.loadmat(taskname)
    task = task["phasenum"]
    task = task[~np.isnan(task).any(axis=1), :]  # Remove all nan rows
    data = mat73.loadmat(filename)
    data = data["clcecog"]
    if os.path.exists(movefile):
        move = mat73.loadmat(movefile)
        move = move["clcmov"]
    print("Data import Complete")
    return task, data, move, elefile


def import_data2(task_folder):
    print("Start Data Import")

    file_fold = task_folder["folder"]
    pat_no = file_fold.split(os.sep)[-2][:4].lower()
    elefile = os.path.join(file_fold, ("gridmap." + pat_no + ".txt"))

    # Load Data and clear data
    """
    task: nparray with task type (col 7), phase time (col 1-5), success or not (col 6)
    data: nparray preprocessed data with 60Hz removal containing n channels
    electrode: Dataframe with Location, Hemisphere, Label,
    """
    taskname = task_folder["data_file"].replace("data.mat", "task.mat")
    movefile = task_folder["data_file"].replace("data.mat", "move.mat")

    move = []

    task = io.loadmat(taskname)
    task = task["phasenum"]
    task = task[~np.isnan(task).any(axis=1), :]  # Remove all nan rows
    data = mat73.loadmat(task_folder["data_file"])
    data = data["clcecog"]
    if os.path.exists(movefile):
        move = mat73.loadmat(movefile)
        move = move["clcmov"]
    print("Data import Complete")
    return task, data, move, elefile


def import_data_baseline(patNo, taskID, cod):
    ## Specify file location
    print("Start Data Import")
    file_loc = "D:\\Blcdata\\output\\"
    patnb = "P" + "0" + str(patNo) + "_"
    patfolder = patnb + taskID
    if cod:
        patfolder = patnb + taskID + "\\" + cod
    file_fold = file_loc + patfolder

    ## Load Data and clear data
    """
    task: nparray with task type (col 7), phase time (col 1-5), success or not (col 6)
    data: nparray preprocessed data with 60Hz removal containing n channels
    electrode: Dataframe with Location, Hemisphere, Label, 
    """
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(
        initialdir=file_fold
    )  # show an "Open" dialog box and return the path to the selected file
    elecnm = "gridmap.p0" + str(patNo) + ".txt"
    elefile = file_fold + "\\" + elecnm

    data = mat73.loadmat(filename)
    data = data["clcecog"]

    print("Data import Complete")
    return data, elefile


def find_task_folders(task_name, base_dir="D:\\Blcdata\\output"):
    task_folders = []
    for folder, subfolders, files in os.walk(base_dir):
        if task_name in folder:
            for subfolder in subfolders:
                subfolder_path = os.path.join(folder, subfolder)
                for file in os.listdir(subfolder_path):
                    if file.endswith("data.mat"):
                        task_folder = {
                            "folder": subfolder_path,
                            "data_file": os.path.join(subfolder_path, file),
                        }
                        task_folders.append(task_folder)

    return task_folders


def create_output_directory(
    patient_name,
    subfolder_name,
    task_name,
    RUN_ID=99,
    output_root="D:\\Blcdata\\results",
):
    # Create the output directory path
    output_dir = os.path.join(
        output_root, str(RUN_ID), str(patient_name), task_name, subfolder_name
    )

    # Check if the output directory already exists
    index = 1
    while os.path.exists(output_dir):
        # Append an index to the directory name if it already exists
        output_dir = os.path.join(
            output_root,
            str(RUN_ID),
            str(patient_name),
            task_name,
            f"{subfolder_name}_{index}",
        )
        index += 1

    # Create the new output directory
    os.makedirs(output_dir)

    return output_dir


def extract_folder_info(task_folder):
    # Get the folder path
    folder_path = task_folder["folder"]

    # Split the folder path
    folder_parts = folder_path.split(os.sep)

    # Extract the patient name, subfolder name, and task name
    patient_name = folder_parts[-2]
    patient_number = int(
        patient_name[2:4]
    )  # Extract the number part and convert to integer
    subfolder_name = folder_parts[-1]
    task_name = folder_parts[-2][5:]

    return patient_number, subfolder_name, task_name


def label_select(taskID):
    phaselabel = []
    if taskID == "Move":
        phaselabel = ["ITI", "Fixation", "Cue1", "Cue2", "Delay", "Response"]
    elif (taskID == "DirectReachGNG") | (taskID == "DirectReach"):
        phaselabel = ["ITI", "Fixation", "Response"]
    elif (taskID == "DelayedReach") | (taskID == "DelayedReachGNG"):
        phaselabel = ["ITI", "Fixation", "Cue1", "Delay", "Response"]
    elif taskID == "DirectReachHold":
        phaselabel = ["ITI", "Fixation", "Response", "Hold"]
    return phaselabel


def trialtime_select(taskname, task, i):
    trialtime = []
    if (
        (taskname == "Move")
        | (taskname == "DirectReach")
        | (taskname == "DelayedReach")
        | (taskname == "DirectReachHold")
    ):
        trialtime = task[i, :-2]
    elif (taskname == "DirectReachGNG") | (taskname == "DelayedReachGNG"):
        trialtime = task[i, :-3]
    return trialtime


def extract_trial(sttime, k, Fs=2000):
    """
    Function to extract phase start and end time
    :param sttime:
    :param k:
    :param taskname:
    :param Fs:
    :return:
    """
    endtime = sttime + 0.5 * Fs
    if k == "ITI":  # ITI
        sttime = int(sttime + 0.5 * Fs)
        endtime = int(sttime + 0.7 * Fs)
    elif k == "Fixation":  # Fix
        sttime = int(sttime + 0.2 * Fs)
        endtime = int(sttime + 1 * Fs)
    elif k == "Cue1":  # Cue1
        sttime = int(sttime + 0.1)
        endtime = int(sttime + 0.7 * Fs)
    elif k == "Cue2":  # Cue2
        sttime = int(sttime + 0.1 * Fs)
        endtime = int(sttime + 0.3 * Fs)
    elif k == "Delay":  # Delay
        sttime = int(sttime + 0.2 * Fs)
        endtime = int(sttime + 0.5 * Fs)
    elif k == "Response":  # Res
        sttime = int(sttime + 0.3 * Fs)
        endtime = int(sttime + 1 * Fs)
    elif k == "Hold":
        sttime = int(sttime + 0.2 * Fs)
        endtime = int(sttime + 1 * Fs)
    return sttime, endtime


def extract_trial_Same_Length(sttime: int, k, dur, Fs=2000):
    """
    Alternative phase selection method to ensure all phases have the same length, which is 1s
    Need to hard code the buffer size
    :param sttime:
    :param Fs:
    :return:
    """

    durf = dur[k]
    buffertime = np.max([(durf - 1 * Fs) / 2, 0.1 * Fs])
    durf = 1 * Fs

    if k == "Response":  # Average 1.25
        buffertime = 0.3 * Fs

    elif k == "ITI":
        buffertime = 0.8 * dur[k]

    elif k == "Delay":
        return int(sttime - 2 * Fs), int(sttime - 1 * Fs)

    return int(sttime + buffertime), int(sttime + durf + buffertime)


def extract_trial_break_delay(sttime: int, k, Fs=2000):
    """
    Alternative phase selection method to combine and break the cue and delay phase into two separate phases
    Cue phase: 0:2s, Delay phase: end-2s:end (Use response index), All other phases: 0.1s buffer + 2s duration
    :param sttime:
    :param k:
    :param dur:
    :param Fs:
    :return:
    """
    durf = 1 * Fs
    if k == "Cue1":
        return int(sttime), int(sttime + durf)
    elif k == "Delay":
        return int(sttime - 0.5 * durf), int(sttime + 0.5 * durf)
    return int(sttime + 0.1 * Fs), int(sttime + durf + 0.1 * Fs)


def extract_trial_GNG(sttime: int, dur, phase, GNG, Fs=2000):
    buffertime = 0.5
    durf = 0
    if phase == "ITI":
        durf = dur[0]
    elif phase == "Fixation":
        durf = dur[1]
    elif phase == "Response":
        if GNG == 1:
            durf = dur[2]
        else:
            durf = dur[3]
        return sttime + int(buffertime * Fs) - durf, sttime - int(
            buffertime * Fs
        )

    return sttime + int(buffertime * Fs // 1), sttime + durf - int(
        buffertime * Fs // 1
    )


def extract_trial_after_ITI(sttime: int, dur, phase, GNG, Fs=2000):
    buffertime = 0.5
    durf = 0
    if phase == "Fixation":
        durf = dur[1]
    elif phase == "ITI":
        sttime = sttime - dur[0] * Fs - buffertime * Fs
    elif phase == "Response":
        if GNG == 1:
            durf = dur[2]
        else:
            durf = dur[3]
        return sttime + int(buffertime * Fs) - durf, sttime - int(
            buffertime * Fs
        )

    return sttime + int(buffertime * Fs // 1), sttime + durf - int(
        buffertime * Fs // 1
    )


def data_select(x1, phasetime, phaselabel, Fs=2000):
    """
    Method only serve to append data in PSD_trial method to combine different phases
    :param x1:
    :param phasetime:
    :param phaselabel:
    :param Fs:
    :return:
    """
    retdata = np.zeros((0, np.shape(x1)[1]))
    for i in range(0, len(phaselabel)):
        if (
            phaselabel[i] == "ITI"
            or phaselabel[i] == "Fixation"
            or phaselabel[i] == "Delay"
        ):
            tepdata = x1[phasetime[i] : phasetime[i] + int(2 * Fs), :]
            retdata = np.vstack((retdata, tepdata))
        elif phaselabel[i] == "Response":
            tepdata = x1[phasetime[i] : phasetime[i] + int(2 * Fs), :]
            retdata = np.vstack((retdata, tepdata))
        elif phaselabel[i] == "Cue1":
            tepdata = x1[phasetime[i] : phasetime[i] + int(1 * Fs), :]
            retdata = np.vstack((retdata, tepdata))
        elif phaselabel[i] == "Cue2":
            tepdata = x1[phasetime[i] : phasetime[i] + int(0.5 * Fs), :]
            retdata = np.vstack((retdata, tepdata))
    retdata = x1[phasetime[0] : phasetime[0] + int(11 * Fs)]
    return retdata


def extract_electrode(electrode, region):
    """

    :param electrode:
    :param region:
    :return:
    """
    if region == "All":
        elearray = electrode
    elif region == "Other":
        elearray = electrode[
            ~(electrode["Location"].str.contains("Hippocampus", case=False))
            & ~(electrode["Location"].str.contains("Amygdala", case=False))
        ]
    elif region == "Both":
        elearray = electrode[
            (electrode["Location"].str.contains("Hippocampus", case=False))
            | (electrode["Location"].str.contains("Amygdala", case=False))
        ]
    else:
        elearray = electrode[
            electrode["Location"].str.contains(region, case=False)
        ]
    elelist = np.zeros(0)
    for i in elearray["GridId"]:
        tep = elearray["Channel"][i]
        elelist = np.hstack((elelist, tep))
    return elelist, elearray


def rewrite_eledata(electrode):
    """

    :param electrode:
    :return:
    """
    eleout = copy.deepcopy(electrode)
    elenum = eleout["Channel"]
    for i in range(0, len(eleout)):
        stele = int(elenum[i].split(":")[0]) - 1
        endele = int(elenum[i].split(":")[-1])
        elesel = np.arange(stele, endele)
        eleout["Channel"][i] = elesel
    return eleout


def rewrite_eledata2(electrode):
    eleout = copy.deepcopy(electrode)
    eleind = 0
    for i in range(0, len(eleout)):
        eleout["Channel"].iloc[i] = (
            electrode["Channel"].iloc[i]
            - electrode["Channel"].iloc[i].min()
            + eleind
        )
        eleind = eleind + len(eleout["Channel"].iloc[i])
    return eleout


def exclude_noise(electrode, noiseind):
    """

    :param electrode:
    :param noiseind:
    :return:
    """
    eleout = copy.deepcopy(electrode)
    for i in range(0, len(electrode)):
        elesel = eleout["Channel"].iloc[i]
        for ind in noiseind:
            elesel = elesel[elesel != ind]
        eleout["Channel"].iloc[i] = elesel
    return eleout


def extract_data(data, region, electrode):
    """

    :param data:
    :param region:
    :param electrode:
    :return:
    """
    clean_ele = rewrite_eledata(
        electrode
    )  # Write the Channel column into ndarray
    elelist, elearray = extract_electrode(clean_ele, region)
    elelist = elelist.astype(np.int64)
    data = data[:, elelist]
    elearray = rewrite_eledata2(elearray)
    return data, elearray


def compress_data(x):
    """

    :param x:
    :return:
    """
    xout = np.array(x[0]).reshape((np.size(x[0]), 1))
    for i in range(1, len(x)):
        xout = np.hstack((xout, np.array(x[i]).reshape((np.size(x[i]), 1))))
    return xout


def Train_Test_Data_Selection(LLtotal, seqidtotal, trainpsd, division):
    totalind = np.arange(0, len(LLtotal) - 1)
    tr_ind = list(set(random.choices(totalind, k=int(division * len(LLtotal)))))
    test_ind = list(set(totalind) - set(tr_ind))

    LLtr = [LLtotal[i] for i in tr_ind]
    controltr = [trainpsd[i] for i in tr_ind]
    seqidtr = [seqidtotal[i] for i in tr_ind]
    LLtest = [LLtotal[i] for i in test_ind]
    controltest = [trainpsd[i] for i in test_ind]
    seqidtest = [seqidtotal[i] for i in test_ind]

    return LLtr, controltr, seqidtr, LLtest, controltest, seqidtest


class Train_Data_Selction:
    tr_ind = []
    test_ind = []
    Tr_Id = []
    Tr_LL = []
    Tr_psd = []

    def __init__(
        self, LLtotal: list, seqidtotal: list, trainpsd: list, division: float
    ):
        self.LLtotal = LLtotal
        self.seqidtotal = seqidtotal
        self.trainpsd = trainpsd
        self.Id_selection(division=division)
        self.Tr_Data_Selection()

    def get_Length(self):
        return len(self.LLtotal)

    def Id_selection(self, division: float):
        length = self.get_Length()
        weight = int(division * length)
        totalind = np.arange(0, length)
        self.tr_ind = list(
            set(random.choices(totalind, k=weight))
        )  # TODO: Change to select unique number
        self.test_ind = list(set(totalind) - set(self.tr_ind))
        while len(self.tr_ind) != weight:
            self.tr_ind = self.tr_ind + list(
                set(random.choices(self.test_ind, k=weight - len(self.tr_ind)))
            )
            self.test_ind = list(set(totalind) - set(self.tr_ind))

    def Duplicate_Data_Helper(self, index: int):
        self.Tr_Id.append(self.seqidtotal[index])
        self.Tr_LL.append(self.LLtotal[index])
        self.Tr_psd.append(self.trainpsd[index])

    def Tr_Data_Selection(self):
        self.Tr_Id = [self.seqidtotal[i] for i in self.tr_ind]
        self.Tr_LL = [self.LLtotal[i] for i in self.tr_ind]
        self.Tr_psd = [self.trainpsd[i] for i in self.tr_ind]

    def Duplicate_Data(self):
        classNumber = np.zeros((0, 1))
        classInd = []
        for i in set(
            self.Tr_Id
        ):  # Transverse all occurance in seqid to find unique classes
            classNumber = np.vstack(
                (classNumber, int(sum(j for j in self.Tr_Id if j == i) / i))
            )  # Add the sum of each class
            classInd.append([j for j in self.tr_ind if self.seqidtotal[j] == i])
        maxClass = int(max(classNumber))
        for i in range(0, len(classNumber)):
            numDup = maxClass - int(classNumber[i])
            for j in range(0, numDup):
                ind = random.choice(classInd[i])
                self.Duplicate_Data_Helper(ind)


def Move_Data_Extract(task, movement, cod):
    Direction = 0
    assert cod in ["Left", "Right"]
    if cod == "Left":
        Direction = 1
    elif cod == "Right":
        Direction = 4
    taskID = set(task[:, -1])
    movedata = []
    for i in taskID:
        tep = []
        for j in range(0, len(task) - 2):
            if task[j, -1] == i:
                tepmove = movement[
                    int(task[j, -3] - 2000 * 0.1) : int(task[j, -3] + 3 * 2000),
                    Direction,
                ]
                tep.append(tepmove)
        movedata.append(tep)
        movedata[i - 1] = np.array(movedata[i - 1]).reshape(
            np.shape(movedata[i - 1])[0], np.shape(movedata[i - 1])[1], 1
        )  # Convert to 3D matrix
    return movedata


def band_to_value(band):
    if band == "alpha":
        return [8, 12]
    elif band == "beta":
        return [13, 30]
    elif band == "gamma":
        return [31, 55]
    else:
        return [75, 100]


def z_norm(data, mean, std):
    """
    Data normalization for each trial
    :param data: N electrode* P phase
    :param mean: N electrode * 1
    :param std: N electrode * 1
    :return: N electrode * P phase
    """
    return (data - mean) / std


def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "n.s"


def bootstrap(data, n_samples):
    """

    :param data: 1D data vector
    :param n_samples: Repitition time
    :return:
    """

    def bootstrap_mean(data):
        indices = np.random.choice(
            data.shape[1], size=data.shape[1], replace=True
        )
        return np.median(data[:, indices], axis=-1)

    # Generate the bootstrap samples and compute the mean of each sample
    bootstrap_means = [bootstrap_mean(data) for _ in range(n_samples)]

    # Compute the 95% confidence interval for the mean using the percentile method
    lower_bound = np.percentile(bootstrap_means, 5, axis=0)
    upper_bound = np.percentile(bootstrap_means, 95, axis=0)

    return lower_bound, upper_bound


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    """

    :param lowcut: low cut frequency
    :param highcut: high cut frequency
    :param fs: sampling frequency
    :param order: filter order
    :return:
    """
    return butter(
        order,
        [lowcut, highcut],
        fs=fs,
        btype="band",
        analog=False,
        output="sos",
    )


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """

    :param data: 1D data vector
    :param lowcut: low cut frequency
    :param highcut: high cut frequency
    :param fs: sampling frequency
    :param order: filter order
    :return:
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def cluster_permutation_test(data1, data2, n_iter, alternative="two-sided"):
    """

    :param data1: 1D matrix n_trials_group_1
    :param data2: 1D matrix n_trials_group_2
    :param n_iter: number of iterations
    :return:
    """
    # Step 0. Compute the observed difference between the two groups
    observed_difference = wilcoxon(
        data1, data2, alternative=alternative, method="approx"
    ).zstatistic
    # Step 1. Combine the two groups into one array
    data = np.concatenate((data1, data2))
    obs_diffs = np.zeros(n_iter)
    for i in range(n_iter):
        # Step 2. Randomly draw as many trials from this combined dataset as there are in each group
        # and assign them to group 1 and group 2
        n_trials_group_1 = data1.shape[0]
        # randomly draw trials from the combined dataset
        group_1_indices = np.random.choice(
            data.shape[0], size=n_trials_group_1, replace=False
        )
        group_2_indices = np.setdiff1d(
            np.arange(data.shape[0]), group_1_indices
        )
        # assign them to group 1 and group 2
        group_1 = data[group_1_indices]
        group_2 = data[group_2_indices]
        # Step 3. Compute the p-value for wilcoxon signrank test
        obs_diffs[i] = wilcoxon(
            group_1, group_2, alternative=alternative, method="approx"
        ).zstatistic
    # Step 4. Compute the proportion of times that
    # the permuted observed difference is larger than the observed differnce
    if alternative == "greater":
        p_value = np.sum(obs_diffs >= observed_difference) / n_iter
    elif alternative == "less":
        p_value = np.sum(obs_diffs <= observed_difference) / n_iter
    else:
        p_value_1 = np.sum(obs_diffs >= observed_difference) / n_iter
        p_value_2 = np.sum(obs_diffs <= observed_difference) / n_iter
        p_value = np.min([p_value_1, p_value_2]) / 2

    return p_value


def cluster_permutation_test_multi_freq(
    psd_cond1, psd_cond2, n_iter=5000, alternative="two-sided"
):
    """
    :param psd_cond1: 2D matrix n_freqs x n_trials_group_1
    :param psd_cond2: 2D matrix n_freqs x n_trials_group_2
    :param n_iter: Number of permutations to perform
    :param alpha: Critical p-value, default is 0.05
    :param alternative: alternative hypothesis, {'two-sided', 'greater', 'less'}
    :return:
    """

    def compute_max_cluster_stat_perm(
        psd_cond1, psd_cond2, alternative="two-sided"
    ):
        """
        :param psd_cond1: 2D matrix n_freqs x n_trials_group_1
        :param psd_cond2: 2D matrix n_freqs x n_trials_group_2
        :return:
        """

        # Randomly permute condition labels
        random_labels = np.random.permutation(
            np.concatenate(
                [np.ones(psd_cond1.shape[1]), np.zeros(psd_cond2.shape[1])]
            )
        )
        psd_perm1 = np.concatenate([psd_cond1, psd_cond2], axis=1)[
            :, random_labels == 1
        ]
        psd_perm2 = np.concatenate([psd_cond1, psd_cond2], axis=1)[
            :, random_labels == 0
        ]

        # Compute test statistic for each frequency bin
        z_stats = wilcoxon(
            psd_perm1,
            psd_perm2,
            axis=1,
            alternative=alternative,
            method="approx",
        ).zstatistic
        num_clusters_perm = 0
        clusters_perm = np.zeros(z_stats.shape)
        if alternative == "two-sided":
            # Identify clusters, 1.96 corresponds to 97.5% percentile
            clusters_perm, num_clusters_perm = label(np.abs(z_stats) >= 1.96)
        elif alternative == "greater":
            # Identify clusters, 1.7 corresponds to 95% percentile
            clusters_perm, num_clusters_perm = label(z_stats >= 1.7)
        elif alternative == "less":
            # Identify clusters, 1.7 corresponds to 95% percentile
            clusters_perm, num_clusters_perm = label(z_stats <= -1.7)
        # Compute cluster-level test statistic
        if num_clusters_perm > 0:
            cluster_stats_perm = np.array(
                [
                    np.sum(z_stats[clusters_perm == i])
                    for i in range(1, num_clusters_perm + 1)
                ]
            )

            # Store the maximum cluster-level test statistic
            return np.max(np.abs(cluster_stats_perm))
        else:
            return 0

    # Compute the test statistic for each frequency bin
    statistic = wilcoxon(
        psd_cond1, psd_cond2, axis=1, alternative=alternative, method="approx"
    )
    test_statistic = statistic.zstatistic

    # Identify clusters based on the test statistic
    clusters, num_clusters = label(np.abs(test_statistic) >= 1.7)
    # Compute cluster-level test statistic
    # Examine if there are clusters
    if num_clusters > 0:
        cluster_stats = np.max(
            np.array(
                [
                    np.sum(test_statistic[clusters == i])
                    for i in range(1, num_clusters + 1)
                ]
            )
        )
    else:
        # If there are no clusters, then two conditions are not significantly different
        p_value = 1
        return p_value

    # Permutation testing
    with Parallel(n_jobs=n_cpus, prefer="threads") as parallel:
        max_cluster_stats_perm = parallel(
            delayed(compute_max_cluster_stat_perm)(psd_cond1, psd_cond2)
            for _ in range(n_iter)
        )

    # Compute p-value
    p_value = 1
    if alternative == "two-sided":
        p_value = (
            1
            - np.sum(np.abs(cluster_stats) >= np.abs(max_cluster_stats_perm))
            / n_iter
        )
    elif alternative == "greater":
        p_value = 1 - np.sum(cluster_stats >= max_cluster_stats_perm) / n_iter
    elif alternative == "less":
        p_value = 1 - np.sum(cluster_stats <= max_cluster_stats_perm) / n_iter

    return p_value


def compute_separate_z_scores(array1, array2):
    """
    Compute Z-Scores for two arrays separately
    1. Combine two arrays
    2. Calculate Z-Scores for combined array along each row
    3. Separate the Z-Scores back into the original array structure
    :param array1:
    :param array2:
    :return:
    """
    # Combine two arrays
    combined = np.concatenate((array1, array2), axis=1)

    # Calculate Z-Scores for combined array along each row
    z_scores_combined = zscore(combined, axis=1)

    # Separate the Z-Scores back into the original array structure
    z_scores_array1 = z_scores_combined[:, : array1.shape[1]]
    z_scores_array2 = z_scores_combined[:, array1.shape[1] :]

    return z_scores_array1, z_scores_array2


def drop_nan(input_list):
    clean_list = [x for x in input_list if not np.isnan(x)]
    return clean_list


def load_pkl_file(patID, taskID, sessionID):
    file_loc = "D:\\Blcdata\\results\\"
    file_path = (
        file_loc + str(sessionID) + "\\" + str(patID) + "\\" + str(taskID)
    )
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(
        initialdir=file_path
    )  # show an "Open" dialog box and return the path to the selected file
    # Load pkl file
    task = pickle.load(open(filename, "rb"))
    file_path = os.path.dirname(filename)
    task.output_dir = file_path + "/"
    print("Data import Complete")
    return task


def yates_z_test(x1, n1, x2, n2):
    p1 = x1 / n1
    p2 = x2 / n2
    p = (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))

    # for two-tailed test
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return z, p_value


def activation_function(data, baseline):
    return 10 * np.log10(data / baseline)


def data_combinination(**kwargs):
    """
    Function to combine data from different bandwidths,
    Data structure should be like data = Channel * [Trials, Class, Time/Frequency bins],
    where first class is the baseline
    After combination, data will be in k array with shape Class * [Trials, Channels*number of input bandwidths]
    :param kwargs: Dictionary with patient number as key and data as value
    :return:
    """
    # Get the first item
    N_channels = np.shape(list(kwargs.values())[0])[0]
    N_trials = np.shape(list(kwargs.values())[0][1])[0]
    N_class = np.shape(list(kwargs.values())[0][1])[1]
    N_band = len(kwargs)
    # Initialize the output as list with length of number of classes
    data_out = []
    for j in range(1, N_class):
        tepT = np.zeros((0, N_channels * N_band))
        for i in range(0, N_trials):
            tep = np.zeros((0))
            for c in range(0, N_channels):
                for k in kwargs.keys():
                    baseline = 10 ** (kwargs[k][c][i, 0, :] / 10)
                    # Examine if data are all 0
                    if np.all(kwargs[k][c][i, j, :] == 0):
                        tep = np.hstack((tep, np.zeros((1))))
                    else:
                        data = 10 ** (kwargs[k][c][i, j, :] / 10)
                        tep = np.hstack(
                            (tep, np.mean(activation_function(data, baseline)))
                        )
            tepT = np.vstack((tepT, tep))
        data_out.append(tepT)
    return data_out


def standarization(training_data):
    scaler = StandardScaler()
    scaler.fit(training_data)
    return scaler


def get_parameter_grid(method):
    param_grid = {}
    if method == "SVM":
        param_grid = {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
            "kernel": ["rbf"],
        }
    elif method == "LDA":
        param_grid = {}
    elif method == "QDA":
        param_grid = {}
    return param_grid


def get_outliers(data):
    """
    Function to get outlier index using IQR method
    :param data:
    :return:
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.where((data > upper_bound) | (data < lower_bound))[0]


def create_index_array(n_dimensions, n_features, n_iter):
    if n_dimensions > n_features:
        raise ValueError(
            "n_dimensions should be less than or equal to n_features"
        )

    index_array = np.zeros((n_iter, n_dimensions), dtype=int)
    selected_combinations = set()

    for i in range(n_iter):
        while True:
            combination = tuple(
                sorted(random.sample(range(n_features), n_dimensions))
            )
            if combination not in selected_combinations:
                selected_combinations.add(combination)
                index_array[i] = combination
                break

    return index_array


if __name__ == "__main__":
    print("Testing Util function")

    # Test the multi-frequency permutation_multi_frequency test

    # Create Data
    np.random.seed(0)
    n_freqs = 20
    trials = 60

    # Create two groups of data
    group1 = np.random.rand(trials, n_freqs) + 20
    group2 = np.random.rand(trials, n_freqs) + 10

    # Plot the raw data
    plt.figure()
    plt.plot(group1.T, color="b", alpha=0.5, label="Group 1")
    plt.plot(group2.T, color="r", alpha=0.5, label="Group 2")
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.legend()
    plt.show()

    # Perform the permutation test
    p_value = cluster_permutation_test_multi_freq(
        group1, group2, n_iter=1000, alternative="less"
    )
