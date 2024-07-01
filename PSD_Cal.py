# Import from global library
import os

import numpy as np
from scipy.signal import detrend
from tqdm import tqdm

# Import from local library
from Data_class import PSDclass
from PSD_cal.hilbert_PSD import choose_wave_hilbert
from PSD_cal.Multi_taper_spectrogram import choose_wave_spec_multitaper
from PSD_cal.multitaper_psd import cal_psd
from PSD_cal.util import choose_wave_psd_range, detrend_data
from PSD_cal.wavelet_spectrogram import choose_wave_wavelet_overtime
from utilfun import (
    extract_trial_after_ITI,
    extract_trial_break_delay,
    extract_trial_GNG,
    extract_trial_Same_Length,
    trialtime_select,
)

n_cpus = os.cpu_count() // 2


#################################SPECTROGRAM_CALCULATION############################################
def Spec_trial(
    x1,
    task,
    movedata,
    cod,
    phaseind,
    method="multitaper",
    Fs=2000,
    detrend_lg=1,
    brainwave=("alpha", "beta", "gamma", "high_gamma"),
    **kwargs,
):
    """

    :param x1:
    :param task:
    :param movedata:
    :param cod:
    :param Fs:
    :param detrend_lg:
    :param brainwave:
    :return:
    """

    LEFT = 2  # Movement electrode: 1 is XL, 4 is XR
    RIGHT = 5
    condition = 0
    assert cod in ["Left", "Right"]
    if cod == "Left":
        condition = LEFT
    elif cod == "Right":
        condition = RIGHT
    if "label" in kwargs:
        label = kwargs["label"]
        phase_name = label[phaseind]
    else:
        phase_name = "All"
    PSDtrial = []
    for i in tqdm(
        range(0, len(task) - 1), desc="Calculating Spectrogram:"
    ):  # Define each trial
        movetype = task[i, -1]
        if phase_name == "All":
            sttime = int(task[i, -3] - 1 * Fs)  # 1s before the movement onset
            endtime = int(task[i, -3] + 3 * Fs)  # 3s after the movement onset
        else:
            if phase_name == "Delay":
                sttime = int(task[i, phaseind + 1])
            else:
                sttime = int(task[i, phaseind])
            sttime, endtime = extract_trial_break_delay(sttime, phase_name)
        totdata = x1[sttime:endtime, :]
        success = task[i, -2]
        if type(movedata) == list and len(movedata) == 0:
            tepmove = movedata
        else:
            tepmove = movedata[sttime:endtime, condition]
        if detrend_lg:
            totdata = detrend_data(totdata)

        if method == "wavelet":
            t = np.arange(
                sttime, endtime, 100
            )  # Dummy time, will be modifed later during plotting
            teppsdclass = PSDclass(i, seqid=movetype, t=t)
            teppsdclass.addmove(tepmove)
            teppsdclass.addraw(totdata)
            teppsdclass1 = choose_wave_wavelet_overtime(
                brainwave, totdata, Fs, teppsdclass, downsample=50
            )
            teppsdclass1.add_success(success)
            PSDtrial.append(teppsdclass1)

        elif method == "multitaper":
            t = np.arange(sttime, endtime, 100)
            teppsdclass = PSDclass(i, seqid=movetype, t=t, ft=np.zeros(0))
            teppsdclass.addmove(tepmove)
            teppsdclass.addraw(totdata)
            teppsdclass1 = choose_wave_spec_multitaper(
                brainwave, totdata, Fs, teppsdclass
            )
            teppsdclass1.add_success(success)
            PSDtrial.append(teppsdclass1)

    print("Calculating spectrogram Complete")
    return PSDtrial


###################################PSD_CALCULATION###################################################
def PSD_cal(
    x1,
    movedata,
    task,
    phasename,
    taskname,
    cod,
    Fs=2000,
    detrend_lg=1,
    debugst=0,
    brainwave=("alpha", "beta", "gamma", "high_gamma"),
    psdmethod="welch",
    phase_time_select="extract_trial_Same_Length",
    **kwargs,
):
    """
    Calculate Single phase PSD for each trial for each frequency band
    :param x1:
    :param movedata:
    :param task:
    :param phasename:
    :param taskname:
    :param cod:
    :param Fs:
    :param detrend_lg:
    :param debugst:
    :param brainwave:
    :param psdmethod:
    :param phase_time_select:
    :param kwargs:
    :return:
    """
    print("PSD calculation using " + psdmethod + " method started")
    task = task.astype(int)
    ind = 0
    psdclass = []
    movetypeset = set(
        task[:, -1]
    )  # To examine movement type: Left, Right, Nothing
    LEFT = 1  # Movement electrode: 1 is XL, 4 is XR
    RIGHT = 4
    condition = 0
    assert cod in ["Left", "Right"]
    if cod == "Left":
        condition = LEFT
    elif cod == "Right":
        condition = RIGHT
    if (
        len(movetypeset) > 3
    ):  # To examine if there is imagined movement involved
        NOTHING = 5
    else:
        NOTHING = 3  # Nothing for movement type
    Badtrial = 1

    for i in tqdm(
        range(0, len(task) - 1), desc="Calculating PSD"
    ):  # Define each trial
        trialtime = trialtime_select(taskname, task, i)
        movetype = task[i, -1]
        if "gng" in kwargs:
            success = task[i, -3]
        else:
            success = task[i, -2]

        for k in range(0, len(trialtime)):  # Define each phase
            phase = phasename[k]

            sttime = trialtime[k]

            if phase == "Delay":
                sttime = trialtime[k + 1]

            if phase_time_select == "extract_trial_Same_Length":
                sttime, endtime = extract_trial_Same_Length(
                    sttime, phase, kwargs["dur"], Fs
                )

            elif (
                phase_time_select == "extract_trial_variable_length"
            ):  # Extract the phase based on preset length
                sttime, endtime = extract_trial_GNG(
                    sttime, kwargs["dur"], phase, kwargs["gng"][i], Fs
                )

            elif phase_time_select == "extract_trial_break_delay":
                sttime, endtime = extract_trial_break_delay(sttime, phase)
            elif phase_time_select == "extract_trial_after_ITI":
                if phase == "ITI":
                    sttime = trialtime[k + 1]
                sttime, endtime = extract_trial_after_ITI(
                    sttime, phase, kwargs["dur"], Fs
                )

            else:  # Extract the entire phase
                if phase == "Response":
                    sttime, endtime = trialtime[k], task[i + 1, 0]
                else:
                    sttime, endtime = trialtime[k], trialtime[k + 1]

            tepdata = x1[sttime:endtime, :]

            if type(movedata) == list and len(movedata) == 0:
                Badtrial = 0
                tepmove = movedata

            else:
                tepmove = movedata[sttime:endtime, condition]

                if detrend_lg:
                    tepmove = detrend(tepmove)

                if phase == "Response" and (
                    (
                        (movetype != NOTHING)
                        and (not np.all(np.abs(tepmove) < 50))
                    )
                    or (
                        (movetype == NOTHING) and (np.all(np.abs(tepmove) < 50))
                    )
                    or (
                        taskname != "Move"
                        and (not np.all(np.abs(tepmove) < 50))
                    )
                ):
                    # Either move during response phase when not in Nothing condition or not move in Nothing condition
                    Badtrial = 0

                if phase != "Response" and np.all(np.abs(tepmove) < 75):
                    # Patient move during other phases
                    Badtrial = 0

                if not np.all(np.abs(tepdata) < 1000):
                    Badtrial = 1

            if not Badtrial or debugst:
                RMS_value = np.sqrt(np.mean(tepdata**2, axis=0))
                if psdmethod == "Hilbert":
                    teppsdclass = PSDclass(i, movetype, phase)
                    teppsdclass1 = choose_wave_hilbert(
                        brainwave, tepdata, Fs, teppsdclass
                    )
                    teppsdclass1.addmove(tepmove)
                    teppsdclass1.addraw(tepdata)
                    teppsdclass1.add_success(success)
                    teppsdclass1.RMS = RMS_value
                    psdclass.append(teppsdclass1)

                elif psdmethod == "multitaper":
                    # Segment will be 1 if data is too short
                    segment = (
                        len(tepdata) // 500
                    )  # Make sure each segment is at most 500=0.25s
                    Sxx, f = cal_psd(tepdata, Fs, n_segment=segment)
                    teppsdclass = PSDclass(i, movetype, phase, f, Sxx=Sxx)
                    teppsdclass1 = choose_wave_psd_range(
                        brainwave, Sxx, f, teppsdclass
                    )
                    teppsdclass1.addmove(tepmove)
                    teppsdclass1.addraw(tepdata)
                    teppsdclass1.RMS = RMS_value
                    teppsdclass1.add_success(success)
                    psdclass.append(teppsdclass1)

                ind = ind + 1
    print("PSD Calculation using  " + psdmethod + "  Complete")
