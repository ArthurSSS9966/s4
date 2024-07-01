import copy
import math
import os
import time
import traceback
import winsound

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedShuffleSplit

from Lee_Lab_Class import Movement, TaskOptions
from Model_training import LSTMClassifier
from models.s4.s4 import S4Block as S4
from utilfun import (
    create_index_array,
    create_output_directory,
    extract_folder_info,
    find_task_folders,
    load_pkl_file,
)

# ================== System Parameters ==================
pd.options.mode.chained_assignment = None
n_cpus = os.cpu_count() // 2
# ================== Parameters ==================
Fs = 2000  # Sampling frequency
region = (  # Brain region to analysis, select Hippocampus, Amygdala, Both, Other or All
    "All"
)
method = (  # DTW method, default is ShiftWarp: only shift the data, does not interpolate the data
    "ShiftWarp"
)
brainwave = ["gamma", "high_gamma"]

# Method choose from 'All_Time', 'extract_trial_Same_Length', 'extract_trial_break_delay'
phase_time_select_method = "extract_trial_break_delay"
referenceMethod = (  # Method choose from 'CMR', 'weighted_CAR', 'CAR'
    "weighted_CAR"
)
# ==================System Parameters ==================
sound_duration = 1000  # milliseconds
sound_freq = 440  # Hz


def run_single(pat_no, subfolder_name, task_name, load_pkl=False, **kwargs):
    """
    Run the analysis on a single task
    :param pat_no:
    :param subfolder_name:
    :param task_name:
    :param load_pkl: Load the data from the pickle file or not
    :return:
    """

    print(
        f"Patient Number: {pat_no}, Subfolder Name: {subfolder_name}, Task"
        f" Name: {task_name}"
    )

    # Create output directory for the task
    OUTPUT_DIR = create_output_directory(
        pat_no, subfolder_name, task_name, RUN_ID
    )
    OUTPUT_DIR = OUTPUT_DIR + "\\"

    # Run DelayedReach analysis on the task
    options = TaskOptions(
        pat_no, subfolder_name, Fs, region, brainwave, OUTPUT_DIR, task_name
    )
    options.phase_time_select_method = phase_time_select_method
    options.referenceMethod = referenceMethod

    if "initialExamine" in kwargs:
        initialExamine = kwargs["initialExamine"]
    else:
        initialExamine = False

    # Load calculation choices
    if "cal_spectrogram" in kwargs:
        cal_spectrogram = kwargs["cal_spectrogram"]
    else:
        cal_spectrogram = False

    task = []
    if load_pkl:
        try:
            task = load_pkl_file(pat_no, task_name, RUN_ID)
            task.clustering()

            if cal_spectrogram:
                task.spectrogram = {}
                task.cal_spectrogram(phase="Response")

            task.save_to_json()

        except Exception as e:
            print("Error in Movement")
            traceback.print_exc()
            print(str(e))

        finally:
            # Play sound when the task is complete
            winsound.Beep(sound_freq, sound_duration)
            return task

    else:
        task = Movement(options)
        if initialExamine:
            try:
                print(task.taskID + " Analysis Starts...")
                task.get_data()
                task.pre_process()
                task.data_quality_check()
                task.save_to_json()

            except Exception as e:
                print("Error in DelayedReach")
                traceback.print_exc()
                print(str(e))

            finally:
                # Play sound when the task is complete
                winsound.Beep(sound_freq, sound_duration)
                return task
        else:
            try:
                print(task.taskID + " Analysis Starts...")
                task.get_data()
                task.pre_process()
                print("Preprocess Done!\nStart Signal Checking...")
                task.data_quality_check()
                print(
                    "Signal Checking done!\nStart PSD calculation and outlier"
                    " removal..."
                )
                task.time_warp(method)
                task.get_duration()
                task.cal_psd_range()
                task.get_outlier()
                task.re_organized_psd_range = task.reorganize_data(
                    data=task.psddata_range
                )

                if cal_spectrogram:
                    task.spectrogram = {}
                    task.cal_spectrogram(phase="Response")

                # Save the class object to jason file
                task.save_to_json()

                print(
                    "PSD calculation and outlier removal done!\nPSD comparison"
                    " in frequency range starts..."
                )

                task.plot_hist_all_channels_all_freqs()

            except Exception as e:
                print("Error in DelayedReach")
                traceback.print_exc()
                print(str(e))
            finally:
                # Play sound when the task is complete
                winsound.Beep(sound_freq, sound_duration)
                return task
