# Import from global library
import copy

import numpy as np
from joblib import Memory
from meegkit import dss
from numpy import matlib
from scipy.stats import zscore

# Import from local library
from Data_class import PSDclass
from Sig_Analysis_Fun import bad_ele_similarity_based, simtable
from utilfun import *


def CLEAN_DATA(
    data,
    electrode,
    region,
    patNo,
    taskID,
    cod,
    Tuned_Parameter=False,
    Save_data=True,
):
    """

    :param data:
    :param electrode:
    :param region:
    :param patNo:
    :param taskID:
    :param cod:
    :param Tuned_Parameter:
    :param Save_data:
    :return:
    """
    file_loc = "D:/Blcdata/output/"
    patnb = "P0" + str(patNo) + "_"
    patfolder = patnb + taskID
    if cod:
        patfolder = patnb + taskID + "/" + cod
    file_fold = file_loc + patfolder
    datafilename = file_fold + "/subdata.npy"
    if os.path.exists(datafilename) and not Tuned_Parameter:
        print("Preprocessed data exists, loading data...")
        with open(str(datafilename), "rb") as f:
            subdata = np.load(f)
        print("Loading Data Done!")
    else:
        print("Data Preprocessing Start...")

        # Clean Line noise
        data = remove_line(data, [60, 100])
        subdata, subelectrode = extract_data(data, region, electrode)

        # Preprocess the data
        denoised_data, eleout = pre_processing(subdata, subelectrode)

        ## Preprocessing cont. (Electrode chosen):
        elearray, _ = extract_electrode(eleout, region)  # choose which dataset
        elearray = elearray.astype(np.int64)
        subdata = denoised_data[:, elearray]

        # subdata,eleout = remove_bad_ele(subdata,eleout)
        print("Data Cleaning Complete")
        if Save_data:
            # Save data
            print("Saving Data...")
            with open(datafilename, "wb") as f:
                np.save(f, subdata)
            print("Saving Data Done!")

    return subdata, eleout


def pre_processing(x1, electrode, art_method="common_avg"):
    """

    :param x1:
    :param electrode:
    :return:
    """
    print("Preprocessing Start")
    # Remove noisy channel and return clean electrode
    xout, noiseind = remove_noise(x1)  # Get noisy channel

    # Rewrite the electrode map
    clean_ele = exclude_noise(
        electrode, noiseind
    )  # Extract useful electrode based on noisy channel

    # artifacts removal
    xout = rm_artifacts(xout, clean_ele, art_method)  # Bipolar rereference

    print("Preprocessing Complete")
    return xout, clean_ele


def remove_bad_ele(subdata, eleout):
    simout = simtable(subdata.T, subdata.T, "cs")
    badele = bad_ele_similarity_based(simout)
    if len(badele) != 0:
        subdata = np.delete(subdata, badele, axis=1)
        baseele = np.concatenate(eleout["Channel"])
        badele = baseele[badele]
        eleout["Channel"] = [
            np.array([x for x in row if x not in badele])
            for row in eleout["Channel"]
        ]
    return subdata, eleout


def remove_noise(x1: np.ndarray):
    """
    Remove bad electrodes based on threshold choosing method:
    1. Mean of amplitude of each electrode must not 1.5 times the overall average of all electrodes
    2. Signal larger than 600uV (Average is around 200-300uV)
    3. Channel with no signals
    4. Electrodes with mean of the amplitude larger than 10,000uV (outliers) are replaced with median in order to calculate
    zscore
    Bad electrode will be replaced with all 0
    :param x1: filtered data
    :return: xout: raw data with 0 replaced in bad electrode; bad electrode number
    xxbad: bad electrode number
    """
    print("Noisy Electrode Rejection Start")
    xmean = np.mean(np.square(x1), axis=0)
    xmean = xmean.astype(int)
    xbad4 = np.where(xmean > 10000)[0]
    xmean[xmean > 10000] = np.median(xmean)
    xbad = np.where(zscore(xmean) > 1.5)[0]
    xbad2 = np.unique(np.where(x1 > 600)[1])
    xbad3 = np.where(xmean == 0)[0]
    xbad = np.hstack((xbad, xbad2, xbad3, xbad4))
    xbad = np.unique(xbad)
    xout = np.array(x1)
    xout[:, xbad] = np.zeros((len(x1), 1))
    print("Noisy Electrode Rejection Complete")
    print("Noisy electrodes are ")
    print(xbad)
    return xout, xbad


def rm_artifacts(x1: np.ndarray, electrode, method: str = "common_avg"):
    """
    Use Common average (Default)/ Bipolar Rereferencing to remove the electrode artifacts
    :param x1: filtered data with bad contacts replaced with 0
    :param electrode: Electrode table
    :param method: common_avg or bipolar
    :return: Rereferenced data
    """
    print("Rereference Start")
    elenum = electrode["Channel"]
    if method == "common_avg":
        xout = np.zeros(np.shape(x1))
        for i in range(0, len(electrode)):
            elesel = electrode.loc[i, "Channel"]
            xout[:, elenum[i]] = x1[:, elesel] - matlib.repmat(
                np.mean(x1[:, elesel], 2), 1, np.size(x1[:, elesel], 2)
            )
        return xout
    if method == "bipolar":
        print("Bipolar rereference start")
        xout = np.zeros(np.shape(x1))
        for i in range(0, len(electrode)):
            elesel = electrode["Channel"].iloc[i]
            for k in range(0, len(elesel)):
                if k != len(elesel) - 1:
                    xout[:, elesel[k]] = x1[:, elesel[k]] - x1[:, elesel[k] + 1]
                else:
                    xout[:, elesel[k]] = x1[:, elesel[k]]
        print("Rereference Complete")
        return xout


def cal_avg(x1, electrode):
    """

    :param x1:
    :param electrode:
    :return:
    """
    y = np.zeros((len(x1), len(electrode)))
    for i in range(0, len(electrode)):
        elesel = electrode["Channel"].iloc[i]
        tepdata = x1[:, elesel]
        y[:, i] = np.mean(tepdata, 1)
    return y


"""
***********************USE MODIFIED VERSION BELOW***********************
"""


def remove_line(x1, lineF, Fs=2000):
    """

    :param x1:
    :param lineF:
    :param Fs:
    :return:
    """
    print("Start Line noise removal")
    xret = np.array(x1)
    for f0 in lineF:
        xret, _ = dss.dss_line_iter(xret, f0, Fs)
    print("Removal Line noise removal Complete")
    return xret


def rm_time_domain_gng(psdclass, E: int, P: int, T: int, phaselabel, gng):
    """

    :param psdclass: psd class calculated from PSD_cal containing all PSD information for each trial, each phase
    :param E: Total number of electrode
    :param P: Total number of phases
    :param T: Total number of trial
    :param phaselabel: Label of each phase
    :return: Time_Outlier: Outlier matrix with [P*E*T] and true indicates is outlier
    """

    def get_outlier_index(data_in, threshold):
        mean_in = np.mean(data_in, axis=1)
        outlier_lower = np.percentile(mean_in, 25)
        outlier_upper = np.percentile(mean_in, 75)
        interrange = outlier_upper - outlier_lower
        outlier_cond1 = (mean_in < outlier_lower - interrange * 1.5) | (
            mean_in > outlier_upper + interrange * 1.5
        )
        outlier_cond2 = np.any(data_in > threshold, axis=1)
        outlier = outlier_cond1 | outlier_cond2
        return outlier

    assert isinstance(psdclass[0], PSDclass)
    threshold = 800  # uV
    Time_Outlier = np.zeros((P, E, T), dtype=bool)
    go_trial = np.where(gng == 1)
    nogo_trial = np.where(gng == 0)
    for p in range(P):
        phase = phaselabel[p]
        for e in range(E):
            tepele = None
            if phase == "Response" or phase == "Delay":
                tepele_nogo = None
                tepoutlier = np.ones((T), dtype=bool)
                for t in range(len(psdclass)):
                    trialid = psdclass[t].trial
                    if psdclass[t].phase == phase:
                        if gng[trialid] == 1:
                            if tepele is None:
                                tepele = np.mean(psdclass[t].get_raw()[:, e])
                            else:
                                tepele = np.vstack(
                                    (
                                        tepele,
                                        np.mean(psdclass[t].get_raw()[:, e]),
                                    )
                                )
                        else:
                            if tepele_nogo is None:
                                tepele_nogo = np.mean(
                                    psdclass[t].get_raw()[:, e]
                                )
                            else:
                                tepele_nogo = np.vstack(
                                    (
                                        tepele_nogo,
                                        np.mean(psdclass[t].get_raw()[:, e]),
                                    )
                                )

                go_outlier = get_outlier_index(tepele, threshold)
                nogo_outlier = get_outlier_index(tepele_nogo, threshold)
                tepoutlier[go_trial] = go_outlier
                tepoutlier[nogo_trial] = nogo_outlier
                assert len(tepoutlier) == T
                Time_Outlier[p, e, :] = tepoutlier

            else:
                for t in range(len(psdclass)):
                    if psdclass[t].phase == phase:
                        if tepele is None:
                            tepele = np.mean(psdclass[t].get_raw()[:, e])
                        else:
                            tepele = np.vstack(
                                (tepele, np.mean(psdclass[t].get_raw()[:, e]))
                            )

                tepoutlier = get_outlier_index(tepele, threshold)
                assert len(tepoutlier) == T
                Time_Outlier[p, e, :] = tepoutlier

    return Time_Outlier


def rm_time_domain(psdclass, E: int, P: int, T: int, phaselabel, **kwargs):
    """

    :param psdclass: psd class calculated from PSD_cal containing all PSD information for each trial, each phase
    :param E: Total number of electrode
    :param P: Total number of phases
    :param T: Total number of trial
    :param phaselabel: Label of each phase
    :param kwargs: movement_type: "Go" or "NoGo" Indicator/ "Left" or "Right" Indicator, etc
    :return: Time_Outlier: Outlier matrix with [P*E*T] and true indicates is outlier
    """
    assert isinstance(psdclass[0], PSDclass)
    threshold = 200  # uV
    Time_Outlier = np.ones((P, E, T), dtype=bool)

    if "movement_type" in kwargs:
        movement_type = kwargs["movement_type"]
        for movement in set(movement_type):
            p = -1  # Only choose the Response phase
            phase = phaselabel[p]
            for e in range(E):
                tepele = []
                success_trials = []
                for t in range(len(psdclass)):
                    if (
                        psdclass[t].phase == phase
                        and psdclass[t].success == 1
                        and psdclass[t].seqid == movement
                    ):
                        tepele.append(np.mean(psdclass[t].get_raw()[e, :]))
                        success_trials.append(psdclass[t].trial)
                success_trials = list(set(success_trials))
                tepele = np.array(tepele)
                Time_Outlier[:, e, list(set(success_trials))] = False
                if len(success_trials) > 10:
                    outlier_lower = np.percentile(tepele, 25)
                    outlier_upper = np.percentile(tepele, 75)
                    interrange = outlier_upper - outlier_lower
                    outlier_cond = (
                        tepele < outlier_lower - interrange * 1.5
                    ) | (tepele > outlier_upper + interrange * 1.5)
                    outlier_cond2 = tepele > threshold
                    outlier = outlier_cond | outlier_cond2
                    assert len(outlier) == len(success_trials)
                    Time_Outlier[p, e, success_trials] = outlier

    else:
        for p in range(P):
            phase = phaselabel[p]
            for e in range(E):
                tepele = []
                success_trials = []
                for t in range(len(psdclass)):
                    if psdclass[t].phase == phase:
                        tepele.append(np.mean(psdclass[t].get_raw()[:, e]))
                        success_trials.append(psdclass[t].success == 1)

                tepele = np.array(tepele)
                success_trials = np.array(success_trials)

                # Skip this electrode if no successful trials
                if not np.any(success_trials):
                    Time_Outlier[p, e, :] = True
                    continue
                # Initialize as False
                outlier_cond1 = np.zeros(T, dtype=bool)
                outlier_cond2 = np.zeros(T, dtype=bool)

                # Compute only for successful trials
                outlier_lower = np.percentile(tepele[success_trials], 25)
                outlier_upper = np.percentile(tepele[success_trials], 75)
                interrange = outlier_upper - outlier_lower

                outlier_cond1[success_trials] = (
                    tepele[success_trials] < outlier_lower - interrange * 1.5
                ) | (tepele[success_trials] > outlier_upper + interrange * 1.5)
                outlier_cond2[success_trials] = np.any(
                    tepele[success_trials] > threshold
                )

                # Combine the conditions
                outlier = outlier_cond1 | outlier_cond2 | (~success_trials)

                assert len(outlier) == T
                Time_Outlier[p, e, :] = outlier

    # # Remove the trial in each electrode with interictal spikes
    # for t in range(len(psdclass)):
    #     phase_index = phaselabel.index(psdclass[t].phase)
    #     trial_index = psdclass[t].trial
    #     for e in range(E):
    #         raw_data = psdclass[t].get_raw()[:, e]
    #         contain_spike = contain_spikes(raw_data)
    #         Time_Outlier[phase_index, e, trial_index] = contain_spike

    return Time_Outlier


def rm_freq_domain_range(
    psdclass, E: int, P: int, T: int, phaselabel, brainwave, **kwargs
):
    """

    :param psdclass: psd class calculated from PSD_cal containing all PSD information for each trial, each phase
    :param E: Total number of electrode
    :param P: Total number of phases
    :param T: Total number of trial
    :param phaselabel: Label of each phase
    :param brainwave: Label of each frequency range of typical brainwave band
    :return: Freq_Outlier: Outlier matrix with size [brainwave] * [Phase * Electrode* Trial]
    """

    assert isinstance(psdclass[0], PSDclass)
    Freq_Outlier = {}  # Create dictionary
    if "movement_type" in kwargs:
        movement_type = kwargs["movement_type"]
        for b in brainwave:
            Freq_Outlier[b] = np.zeros((P, E, T), dtype=bool)
            for movement in set(movement_type):
                for p in range(P):
                    phase = phaselabel[p]
                    for e in range(E):
                        tepele = []
                        success_trials = []
                        for t in range(len(psdclass)):
                            if (
                                psdclass[t].phase == phase
                                and psdclass[t].success == 1
                                and psdclass[t].seqid == movement
                            ):
                                tepele.append(
                                    np.mean(psdclass[t].getband(b)[e, :])
                                )
                                success_trials.append(psdclass[t].trial)
                        success_trials = list(set(success_trials))
                        Freq_Outlier[b][p, e, success_trials] = False
                        if len(success_trials) > 10:
                            outlier_lower = np.percentile(tepele, 25)
                            outlier_upper = np.percentile(tepele, 75)
                            interrange = outlier_upper - outlier_lower
                            outlier_cond = (
                                tepele < outlier_lower - interrange * 1.5
                            ) | (tepele > outlier_upper + interrange * 1.5)
                            Freq_Outlier[b][p, e, success_trials] = outlier_cond

    else:
        for b in brainwave:
            Freq_Outlier[b] = np.zeros((P, E, T), dtype=bool)
            for p in range(P):
                phase = phaselabel[p]
                for e in range(E):
                    tepele = []
                    success_trials = []
                    for t in range(len(psdclass)):
                        if (
                            psdclass[t].phase == phase
                            and psdclass[t].success == 1
                        ):
                            tepele.append(np.mean(psdclass[t].getband(b)[e, :]))
                            success_trials.append(psdclass[t].trial)
                    success_trials = list(set(success_trials))
                    if len(success_trials) > 0:
                        outlier_lower = np.percentile(tepele, 25)
                        outlier_upper = np.percentile(tepele, 75)
                        interrange = outlier_upper - outlier_lower
                        outlier_cond = (
                            tepele < outlier_lower - interrange * 1.5
                        ) | (tepele > outlier_upper + interrange * 1.5)
                        Freq_Outlier[b][p, e, success_trials] = outlier_cond
                    Freq_Outlier[b][
                        p, e, list(set(range(T)) - set(success_trials))
                    ] = True
    return Freq_Outlier


def rm_freq_domain_range_gng(
    psdclass, E: int, P: int, T: int, phaselabel, brainwave, gng
):
    """

    :param psdclass: psd class calculated from PSD_cal containing all PSD information for each trial, each phase
    :param E: Total number of electrode
    :param P: Total number of phases
    :param T: Total number of trial
    :param phaselabel: Label of each phase
    :param brainwave: Label of each frequency range of typical brainwave band
    :return: Freq_Outlier: Outlier matrix with size [brainwave] * [Phase * Electrode* Trial]
    """

    def get_outlier_index(data_in):
        mean_in = np.mean(data_in, axis=1)
        outlier_lower = np.percentile(mean_in, 25)
        outlier_upper = np.percentile(mean_in, 75)
        interrange = outlier_upper - outlier_lower
        outlier = (mean_in < outlier_lower - interrange * 1.5) | (
            mean_in > outlier_upper + interrange * 1.5
        )
        return outlier

    assert isinstance(psdclass[0], PSDclass)
    go_trial = np.where(gng == 1)
    nogo_trial = np.where(gng == 0)
    Freq_Outlier = {}  # Create cell array
    for b in brainwave:
        Freq_Outlier[b] = np.zeros((P, E, T), dtype=bool)
        for p in range(P):
            phase = phaselabel[p]
            for e in range(E):
                tepele = None
                if phase == "Response" or phase == "Delay":
                    tepele_nogo = None
                    tepoutlier = np.ones((T), dtype=bool)
                    for t in range(len(psdclass)):
                        trialid = psdclass[t].trial
                        if psdclass[t].phase == phase:
                            if gng[trialid] == 1:
                                if tepele is None:
                                    tepele = np.mean(
                                        psdclass[t].getband(b)[e, :]
                                    )
                                else:
                                    tepele = np.vstack(
                                        (
                                            tepele,
                                            np.mean(
                                                psdclass[t].getband(b)[e, :]
                                            ),
                                        )
                                    )
                            else:
                                if tepele_nogo is None:
                                    tepele_nogo = np.mean(
                                        psdclass[t].getband(b)[e, :]
                                    )
                                else:
                                    tepele_nogo = np.vstack(
                                        (
                                            tepele_nogo,
                                            np.mean(
                                                psdclass[t].getband(b)[e, :]
                                            ),
                                        )
                                    )

                    go_outlier = get_outlier_index(tepele)
                    nogo_outlier = get_outlier_index(tepele_nogo)
                    tepoutlier[go_trial] = go_outlier
                    tepoutlier[nogo_trial] = nogo_outlier
                    assert len(tepoutlier) == T
                    Freq_Outlier[b][p, e, :] = tepoutlier

                else:
                    for t in range(len(psdclass)):
                        if psdclass[t].phase == phase:
                            if tepele is None:
                                tepele = np.mean(psdclass[t].getband(b)[e, :])
                            else:
                                tepele = np.vstack(
                                    (
                                        tepele,
                                        np.mean(psdclass[t].getband(b)[e, :]),
                                    )
                                )

                    tepoutlier = get_outlier_index(tepele)
                    assert len(tepoutlier) == T
                    Freq_Outlier[b][p, e, :] = tepoutlier
    return Freq_Outlier


def rm_bad_trial(time_outlier, freq_outlier, task):
    """
    Make the entire trial outlier trial in any phase in the trial is an outlier
    :param time_outlier:
    :param freq_outlier:
    :return: modified_time_outlier : Same dimension as time_outlier but with entire trial of certain electrode removed
    :return: modified_freq_outlierSame dimension as freq_outlier but with entire trial of certain electrode removed
    of all frequency band
    """

    modified_freq_outlier = copy.deepcopy(freq_outlier)
    modified_time_outlier = copy.deepcopy(time_outlier)

    P, E, T = np.shape(time_outlier)
    B = list(freq_outlier.keys())

    for i in range(E):
        for j in range(T):
            response_time = (task[j + 1, 0] - task[j, -3]) / 2000
            if response_time < 0.5:
                modified_time_outlier[:, i, j] = True
            if any(time_outlier[:, i, j]):
                modified_time_outlier[:, i, j] = True

    for b in B:
        for i in range(E):
            for j in range(T):
                if any(freq_outlier[b][:, i, j]):
                    modified_freq_outlier[b][:, i, j] = True

    return modified_time_outlier, modified_freq_outlier


def rm_bad_list(psdclass, phaselabel, electrode, task, brainwave, **kwargs):
    """
    :param psdclass:
    :param phaselabel:
    :param electrode:
    :param task:
    :param brainwave:
    :param gng:
    :return: mod_time_outlier: [P* E* T] boolean array indicating which trial is outlier, if a single P is outlier,
    all P in that T will become outliers.
    :return: mod_freq_outlier: [B] * [P* E* T] boolean array similar as mod_time_outlier array
    """

    P = len(phaselabel)
    E = electrode["Channel"][len(electrode) - 1][-1] + 1
    T = len(psdclass) // P

    time_outlier = rm_time_domain(psdclass, E, P, T, phaselabel, **kwargs)
    freq_outlier = rm_freq_domain_range(
        psdclass, E, P, T, phaselabel, brainwave, **kwargs
    )

    mod_time_outlier, mod_freq_outlier = rm_bad_trial(
        time_outlier, freq_outlier, task
    )

    return mod_time_outlier, mod_freq_outlier


def rm_bad_list_gng(psdclass, phaselabel, electrode, task, brainwave, gng):
    """
    :param psdclass:
    :param phaselabel:
    :param electrode:
    :param task:
    :param brainwave:
    :param gng:
    :return: mod_time_outlier: [P* E* T] boolean array indicating which trial is outlier, if a single P is outlier,
    all P in that T will become outliers.
    :return: mod_freq_outlier: [B] * [P* E* T] boolean array similar as mod_time_outlier array
    """

    P = len(phaselabel)
    E = electrode["Channel"][len(electrode) - 1][-1] + 1
    T = len(task) - 1

    time_outlier = rm_time_domain_gng(psdclass, E, P, T, phaselabel, gng)
    freq_outlier = rm_freq_domain_range_gng(
        psdclass, E, P, T, phaselabel, brainwave, gng
    )

    mod_time_outlier, mod_freq_outlier = rm_bad_trial(
        time_outlier, freq_outlier, task
    )

    assert np.shape(mod_time_outlier) == (P, E, T)

    return mod_time_outlier, mod_freq_outlier


def calculate_weights(rms_values):
    def median_absolute_deviation(values):
        median = np.median(values)
        return np.median(np.abs(values - median))

    def gaussian_function(x, median, mad):
        return np.exp(-((x - median) ** 2) / (2 * mad**2))

    weights = -1 * rms_values + 2 * np.max(abs(rms_values))

    # median_rms = np.median(rms_values)
    # mad_rms = median_absolute_deviation(rms_values)
    # weights = gaussian_function(rms_values, median_rms, mad_rms)

    # Normalize the weights so that they sum up to 1
    weights = weights / np.sum(weights)

    return weights


def apply_CAR(data, electrodes, method="median"):
    """
    data: A NumPy array with dimensions (N * K, T), where T is the number of time points.
    electrodes: The number of electrodes, N.
    channels_per_electrode: The number of channels per electrode, K.
    """
    re_referenced_data = np.empty_like(data)

    for i in range(len(electrodes)):
        electrode_data = data[:, electrodes[i]]
        filtered_raw_data = butter_highpass_filter(
            electrode_data.T, 30, fs=2000, order=5
        )
        filtered_raw_data = filtered_raw_data.T
        # Delete any channels with amplitude larger than 20uV and copy the data only with good channel to a new array
        cleaned_electrode_data = electrode_data[
            :, np.all(np.abs(filtered_raw_data) < 20, axis=0)
        ]

        # if cleaned_electrode_data has at least 2 channels
        if cleaned_electrode_data.shape[1] > 1:
            if method == "median":
                common_average = np.median(cleaned_electrode_data, axis=1)
            elif method == "weighted_mean":
                # Calculate RMS for each electrode (axis=0 to operate along columns)
                rms_values = np.sqrt(
                    np.mean(cleaned_electrode_data**2, axis=0)
                )
                weights = calculate_weights(rms_values)
                common_average = np.average(
                    cleaned_electrode_data, axis=1, weights=weights
                )
            else:
                common_average = np.mean(cleaned_electrode_data, axis=1)

            re_referenced_data[:, electrodes[i]] = (
                electrode_data - common_average[:, np.newaxis]
            )
        else:
            re_referenced_data[:, electrodes[i]] = electrode_data

    return re_referenced_data


def contain_spikes(raw_data, threshold=20):
    filtered_raw_data = butter_highpass_filter(raw_data.T, 30, fs=2000, order=5)
    filtered_raw_data = filtered_raw_data.T
    # Return true if there a time point with amplitude larger than 20uV
    return np.any(np.abs(filtered_raw_data) > threshold)
