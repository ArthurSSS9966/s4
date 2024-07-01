import copy
import math
import os
import time
import traceback
import winsound

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedShuffleSplit

from Lee_Lab_Class import (
    Baseline,
    DelayedReach,
    DelayedReachGNG,
    DirectReachGNGV2,
    Movement,
    TaskOptions,
)
from Model_training import LSTMClassifier
from utilfun import (
    create_index_array,
    create_output_directory,
    extract_folder_info,
    find_task_folders,
    load_pkl_file,
)

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
brainwave = ["beta", "gamma"]

# Method choose from 'All_Time', 'extract_trial_Same_Length', 'extract_trial_break_delay'
phase_time_select_method = "extract_trial_break_delay"
referenceMethod = (  # Method choose from 'CMR', 'weighted_CAR', 'CAR'
    "weighted_CAR"
)
# ==================System Parameters ==================
sound_duration = 1000  # milliseconds
sound_freq = 440  # Hz


def run_batch(task_name, RUN_ID, start_patient=0):
    """
    Run the analysis on all files with the desired task
    :param task_name:
    :param RUN_ID:
    :return:
    """

    if RUN_ID != 99:
        save = True
    else:
        save = False

    # Loop over all task folders
    task_folders = find_task_folders(task_name)

    for task_folder in task_folders:
        # Compute the time of each iteration
        start_time = time.time()

        print(
            f"Folder: {task_folder['folder']}, Data file:"
            f" {task_folder['data_file']}"
        )

        pat_no, subfolder_name, task_name = extract_folder_info(task_folder)
        if pat_no < start_patient:
            continue
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
        options.task_folder = task_folder
        options.referenceMethod = referenceMethod
        try:
            if task_name == "DirectReachGNG" or task_name == "DelayedReachGNG":
                direct_reach_gng = DirectReachGNGV2(options)
                direct_reach_gng.excute_Direct_Reach_GNG()
            elif (task_name == "DelayedReach") or (task_name == "DirectReach"):
                delayed_reach = DelayedReach(options)
                delayed_reach.excute_Delayed_Reach(save=save)

            print(task_name + " Complete")
        except Exception as e:
            # Catch the error
            print("Error in " + task_name)
            print(e)
            continue

        # Compute the time of each iteration
        end_time = time.time()
        print(f"Time: {end_time - start_time} seconds")
        print("=========================================")
        winsound.Beep(sound_freq, sound_duration)


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

    if task_name == "Move":
        task = []
        if load_pkl:
            if initialExamine:
                task = Movement(options)
                try:
                    print(task.taskID + " Analysis Starts...")
                    task = load_pkl_file(pat_no, task_name, RUN_ID)
                    task.data_quality_check()
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
                        "Signal Checking done!\nStart PSD calculation and"
                        " outlier removal..."
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
                        "PSD calculation and outlier removal done!\nPSD"
                        " comparison in frequency range starts..."
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


def parallel_computing_accuracy(task, band, N_channels, shuffling=False):
    """
    Function for parallel computing the accuracy map
    :param task:
    :param band:
    :param N_channels:
    :return:
    """
    start_time = time.time()
    accuracy_map = np.zeros((N_channels, len(band)))
    rho_map = np.zeros((N_channels, len(band)))
    var_map = np.zeros((N_channels, len(band)))
    for b in range(len(band)):
        key, value = list(task.re_organized_psd_range.items())[b]
        data = task.clustering({key: value})
        for c in range(N_channels):
            # Combine the c column and the last column into a new array
            dataset = np.concatenate(
                (data[:, c].reshape(-1, 1), data[:, -1].reshape(-1, 1)), axis=1
            )
            if shuffling == True:
                # Shuffle the label
                np.random.shuffle(dataset[:, -1])

            # Select only the data with class label 1,2,5
            dataset = dataset[np.isin(dataset[:, -1], [0, 1, 4])]
            # Run a correlation analysis on the feature and the label
            rho, pval = spearmanr(dataset[:, 0], dataset[:, 1])
            rho_map[c, b] = np.abs(rho)
            # Compute the accuracy
            accuracy_map[c, b] = task.train_test_dataset(dataset)
            # Compute the variance of the feature
            var = np.var(dataset[:, 0])
            var_map[c, b] = var
    print(f"Time for each iteration: {time.time() - start_time} seconds")
    return accuracy_map, rho_map, var_map


def parallel_computing_accuracy_N_dimensions(
    task, n_dimensions="all", band="beta", model="SVM"
):
    """
    Function for parallel computing the accuracy for neuron dropping curve
    :param n_dimensions: Number of features fed into train_test_dataset
    :param task:
    :return:
    """
    start_time = time.time()
    bandindex = {"alpha": 0, "beta": 1, "gamma": 2, "high_gamma": 3}
    # Get the index of the band
    ind = bandindex[band]

    if band == "all":
        data = task.clustering()
    else:
        key, value = list(task.re_organized_psd_range.items())[ind]
        data = task.clustering({key: value})

    # Plot correlation matrix of the data
    corr = np.corrcoef(data[:, :-1].transpose())
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.title(f"Correlation Matrix of the data for band {band}")
    plt.show()

    # Delete the channel with correlation coefficient larger than 0.9
    corr = np.abs(corr)
    corr = corr - np.eye(corr.shape[0])
    corr = np.where(corr > 0.9)
    corr = np.unique(corr[0])
    data = np.delete(data, corr, axis=1)

    label = data[:, -1]
    data = data[:, :-1]

    if n_dimensions == "all":
        n_iter = 1
    else:
        n_dimensions = int(n_dimensions)
        n_iter = int(
            np.min(
                [
                    200 * n_dimensions,
                    10000,
                    math.comb(np.shape(data)[1], n_dimensions),
                ]
            )
        )
    accuracy_map = np.zeros(n_iter)

    if n_dimensions != "all":
        # Create a random but no same index array with
        # row as the number of iterations and
        # column as the number of dimensions
        random_index = create_index_array(
            n_dimensions, np.shape(data)[1], n_iter
        )
    else:
        random_index = np.array([np.arange(data.shape[1])])

    for n in range(n_iter):
        if n_dimensions != "all":
            index = random_index[n]
        else:
            index = random_index
        tep_data = data[:, index]
        # Combine the c column and the last column into a new array
        dataset = np.concatenate((tep_data, label.reshape(-1, 1)), axis=1)

        # # Select only the data with class label 0，1，4
        # dataset = dataset[np.isin(dataset[:, -1], [0, 1, 4])]

        # Compute the accuracy
        accuracy_map[n] = task.train_test_dataset(dataset, model=model)

    print(f"Time for each iteration: {time.time() - start_time} seconds")
    return accuracy_map, random_index, corr


def train_test_model(
    task, dPCA=True, random_state=2023, model="LSTM", **kwargs
):
    """
    Single run of algorithm for training and testing different models and return the accuracy
    :param dPCA:
    :param rebalance:
    :param random_state:
    :return:
    """
    band_type = kwargs.get("band_type", "All")
    region = kwargs.get("region", "All")
    if dPCA:
        n_components = kwargs.get("n_components", 10)
        test_data, label = task.dPCA_transform(
            datatype="psd",
            sig_mask=False,
            n_component=n_components,
            band=band_type,
            region=region,
        )

    else:
        task._construct_dPCA_dataset(
            datatype="psd",
            task_to_discriminate="Response",
            shuffle=False,
            dur=1.5,
            band=band_type,
            region=region,
        )
        # Combine the test data into a single array
        test_data = np.concatenate(
            (task.dpca_data[0], task.dpca_data[1], task.dpca_data[2]), axis=1
        )
        # Create the label array
        label = np.concatenate(
            (
                np.zeros(task.dpca_data[0].shape[1]),
                np.ones(task.dpca_data[1].shape[1]),
                np.ones(task.dpca_data[2].shape[1]) * 2,
            ),
            axis=0,
        )
        test_data = test_data.transpose(1, 2, 0)

    n_trials = [np.sum(task.movement_type == i) for i in range(1, 4)]
    # rearrange the test_data and combine three class with only the first dimension
    original_data_to_plot = test_data[:, :, 0]
    # Plot the original data and first three trials of the original data in a separate subplot
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(original_data_to_plot, aspect="auto")
    # Add horizontal lines to separate the three classes
    ax1.hlines(
        n_trials[0],
        0,
        original_data_to_plot.shape[1],
        colors="r",
        linestyles="dashed",
    )
    ax1.hlines(
        n_trials[0] + n_trials[1],
        0,
        original_data_to_plot.shape[1],
        colors="r",
        linestyles="dashed",
    )
    ax1.set_title(
        "Transformed data for" + band_type + " band" + region + " region"
    )
    ax2.plot(
        np.mean(original_data_to_plot[0 : n_trials[0], :], axis=0),
        label="Right",
    )
    ax2.plot(
        np.mean(
            original_data_to_plot[
                n_trials[0] + 1 : n_trials[0] + n_trials[1] + 1, :
            ],
            axis=0,
        ),
        label="Left",
    )
    ax2.plot(
        np.mean(
            original_data_to_plot[n_trials[0] + n_trials[1] + 2 : -1, :], axis=0
        ),
        label="Nothing",
    )
    ax2.legend(fontsize=10)
    ax2.set_title("First three trials of the transformed data", fontsize=10)
    plt.show()

    accuracies = []

    if model == "LSTM":
        epoches = kwargs.get("epoches", 100)
        # Shuffle and split data and label
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=random_state
        )
        for train_index, test_index in sss.split(test_data, label):
            label_train, label_test = label[train_index], label[test_index]
            test_data_train, test_data_test = (
                test_data[train_index],
                test_data[test_index],
            )
        model = LSTMClassifier(
            input_dim=test_data_train.shape[2], output_dim=3, lr=0.001
        )
        model.random_init()
        model.load_data(
            test_data_train, test_data_test, label_train, label_test
        )
        accuracies, _, _ = model.train_test_using_optimizer(epoches=epoches)
        accuracies = np.max(accuracies)

    elif model == "Chance":
        # Randomly assign the test label and compute the accuracy
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=random_state
        )
        for train_index, test_index in sss.split(test_data, label):
            label_train, label_test = label[train_index], label[test_index]
            class_number = np.unique(label_test)
            for i in range(100):
                shuffled_label = np.random.choice(
                    class_number, size=len(label_test), replace=True
                )
                accuracies.append(
                    np.sum(label_test == shuffled_label) / len(label_test)
                )

        accuracies = np.mean(accuracies)

    else:
        # Take the avearge of each trial
        test_data_avg = np.mean(test_data, axis=1)
        test_data_combined = np.concatenate(
            (test_data_avg, label.reshape(-1, 1)), axis=1
        )
        accuracies = task.train_test_dataset(
            test_data_combined,
            model=model,
            random_state=random_state,
            n_fold=10,
        )

    return accuracies


if __name__ == "__main__":
    taskID = "Move"  # Task type
    pat_no = 64  # Patient number
    cod = "Right"  # Condition type Left hand or Right hand
    load = True  # Load the data from the pickle file or not
    initialExamine = False
    model = (  # Model choose from "LSTM&dPCA", "QDA", "Compare_xxx", "None"
        "Compare_region"
    )
    RUN_ID = 99  # 99 for debugging purpose
    cal_spectrogram = False
    plot_spectrogram = False
    plot_comparison = False
    plotPSDRange = False
    plotRegion = "All"

    # run_batch(taskID, RUN_ID, pat_no)

    if taskID != "Move" or load is False:
        task = run_single(
            pat_no,
            cod,
            taskID,
            load_pkl=load,
            initialExamine=initialExamine,
            cal_spectrogram=cal_spectrogram,
            plot_spectrogram=plot_spectrogram,
            plot_comparison=plot_comparison,
            plotPSDRange=plotPSDRange,
            plotRegion=plotRegion,
        )

    ###########################Explorative steps################################
    if load is True and taskID == "Move":
        task = run_single(
            pat_no, cod, taskID, load_pkl=load, cal_spectrogram=cal_spectrogram
        )

        if model == "LSTM&dPCA":
            task.dPCA(datatype="psd", sig_mask=False)

            # Combine the test data into a single array
            test_data = np.concatenate(
                (task.dpca_data[0], task.dpca_data[1], task.dpca_data[2]),
                axis=1,
            )
            # Create the label array
            label = np.concatenate(
                (
                    np.zeros(task.dpca_data[0].shape[1]),
                    np.ones(task.dpca_data[1].shape[1]),
                    np.ones(task.dpca_data[2].shape[1]) * 2,
                ),
                axis=0,
            )

            test_data_transformed = task.dpca.transform(test_data)["st"]
            test_data_transformed = test_data_transformed.transpose(1, 2, 0)
            test_data = test_data.transpose(1, 2, 0)

            # rearrange the test_data and combine three class with only the first dimension
            test_data_to_plot = test_data_transformed[:, :, 0]
            original_data_to_plot = test_data[:, :, 0]

            # Plot the transformed data and first three trials of the transformed data in a separate subplot
            _, (ax1, ax2) = plt.subplots(2, 1)
            ax1.imshow(test_data_to_plot, aspect="auto")
            ax1.set_title("Transformed data")
            ax2.plot(test_data_to_plot[0, :], label="Right")
            ax2.plot(test_data_to_plot[28, :], label="Left")
            ax2.plot(test_data_to_plot[-1, :], label="Nothing")
            ax2.legend(fontsize=10)
            ax2.set_title(
                "First three trials of the transformed data", fontsize=10
            )
            plt.show()

            # Plot the original data and first three trials of the original data in a separate subplot
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.imshow(original_data_to_plot, aspect="auto")
            ax1.set_title("Original data")
            ax2.plot(original_data_to_plot[0, :], label="Right")
            ax2.plot(original_data_to_plot[28, :], label="Left")
            ax2.plot(original_data_to_plot[-1, :], label="Nothing")
            ax2.legend(fontsize=10)
            ax2.set_title(
                "First three trials of the original data", fontsize=10
            )
            plt.show()

            # LSTM Model training and testing
            (
                test_data_train,
                test_data_transformed_train,
                label_train,
                label_test,
                test_data_transformed_test,
                test_data_test,
            ) = ([], [], [], [], [], [])
            # Shuffle and split data and label
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=20230923
            )
            for train_index, test_index in sss.split(
                test_data_transformed, label
            ):
                test_data_transformed_train, test_data_transformed_test = (
                    test_data_transformed[train_index],
                    test_data_transformed[test_index],
                )
                label_train, label_test = label[train_index], label[test_index]
                test_data_train, test_data_test = (
                    test_data[train_index],
                    test_data[test_index],
                )

            num_models = 20
            models = [
                LSTMClassifier(
                    input_dim=test_data_transformed_train.shape[2],
                    output_dim=3,
                    lr=0.01,
                    hidden_dim=i * 2,
                )
                for i in range(1, num_models)
            ]

            trans_accuracies_total = []
            transformed_accuracies = []
            transformed_recall = []
            transformed_f1 = []

            for i, model in enumerate(models):
                model.random_init()
                model.load_data(
                    test_data_transformed_train,
                    test_data_transformed_test,
                    label_train,
                    label_test,
                )
                (
                    transformed_accuracies,
                    transformed_recall,
                    transformed_f1,
                ) = model.train_test_using_optimizer(epoches=200)
                trans_accuracies_total.append(transformed_accuracies[-1])

            # LSTM Model training and testing untransformed data
            normal_models = [
                LSTMClassifier(
                    input_dim=test_data_train.shape[2],
                    output_dim=3,
                    lr=0.01,
                    hidden_dim=i * 2,
                )
                for i in range(1, num_models)
            ]

            normal_accuracies_total = []
            untrans_accuracies = []
            untrans_recall = []
            untrans_f1 = []

            for i, normal_model in enumerate(normal_models):
                normal_model.random_init()
                normal_model.load_data(
                    test_data_train, test_data_test, label_train, label_test
                )
                (
                    untrans_accuracies,
                    untrans_recall,
                    untrans_f1,
                ) = normal_model.train_test_using_optimizer(epoches=200)
                normal_accuracies_total.append(untrans_accuracies[-1])

            # Plot accuracy over epochs
            plt.plot(transformed_accuracies, "r-", label="Transformed Data")
            plt.plot(untrans_accuracies, "b-", label="Original Data")
            plt.xlabel("Epoch")
            plt.ylabel("Test Accuracy")
            plt.title("Test Accuracy for data Over Epochs")
            plt.legend()
            plt.show()

            # Plot the accuracy of trans_accuracies_total and normal_accuracies_total
            plt.figure()
            plt.plot(trans_accuracies_total, "r-", label="Transformed Data")
            plt.plot(normal_accuracies_total, "b-", label="Original Data")
            plt.xlabel("Model_initialization")
            plt.ylabel("Test Accuracy")
            plt.title("Test Accuracy for data Over Model_initialization")
            plt.legend()
            plt.show()

        elif model == "None":
            pass

        elif model == "Compare_models":
            n_iter = 10
            LSTM_trans_accuracies = []
            LSTM_untrans_accuracies = []
            LDA_trans_accuracies = []
            LDA_untrans_accuracies = []
            for i in range(n_iter):
                LSTM_trans_accuracies.append(
                    train_test_model(
                        task,
                        dPCA=True,
                        random_state=20231002 + i,
                        model="LSTM",
                        n_components=20,
                    )
                )
                LSTM_untrans_accuracies.append(
                    train_test_model(
                        task,
                        dPCA=False,
                        random_state=20231002 + i,
                        model="LSTM",
                    )
                )
                LDA_trans_accuracies.append(
                    train_test_model(
                        task,
                        dPCA=True,
                        random_state=20231002 + i,
                        model="QDA",
                        n_components=20,
                    )
                )
                LDA_untrans_accuracies.append(
                    train_test_model(
                        task, dPCA=False, random_state=20231002 + i, model="QDA"
                    )
                )

            data_to_plot = [
                LSTM_trans_accuracies,
                LSTM_untrans_accuracies,
                LDA_trans_accuracies,
                LDA_untrans_accuracies,
            ]

            # Create the boxplot
            fig, ax = plt.subplots()
            ax.boxplot(data_to_plot)

            # Add titles and labels
            ax.set_title("Model Accuracy Comparison")
            ax.set_xlabel("Models")
            ax.set_ylabel("Accuracy")
            ax.set_xticklabels(
                ["dPCA+LSTM", "LSTM only", "dPCA+LDA", "LDA only"]
            )
            # Show the plot
            plt.show()

        elif model == "Compare_band":
            n_iter = 10
            bands = ["alpha", "beta", "gamma", "high_gamma"]
            LSTM_trans_accuracies = []
            for band in bands:
                for i in range(n_iter):
                    print(f"Computing the accuracy for {band} band...")
                    LSTM_trans_accuracies.append(
                        train_test_model(
                            task,
                            dPCA=True,
                            random_state=20231002 + i,
                            model="LSTM",
                            n_components=20,
                            band_type=band,
                            epoches=100,
                        )
                    )

            # Rearrange the LSTM_trans_accuracies
            LSTM_trans_accuracies = np.reshape(
                LSTM_trans_accuracies, (len(bands), n_iter)
            )
            # Create the boxplot
            fig, ax = plt.subplots()
            ax.boxplot(LSTM_trans_accuracies.T)
            # Set ylim
            ax.set_ylim([0.3, 1])

            # Add titles and labels
            ax.set_title("Model Accuracy Comparison")
            ax.set_xlabel("Band")
            ax.set_ylabel("Accuracy")
            ax.set_xticklabels(bands)
            # Show the plot
            plt.show()

        elif model == "Compare_region":
            n_iter = 10
            regions = task.new_ele["Label"].unique().tolist()
            # append all to region
            regions.append("All")
            LSTM_trans_accuracies = []
            pure_chance = []
            regions_to_plot = copy.deepcopy(regions)
            for region in regions:
                # Check if region has at least 2 electrode
                if region != "All":
                    channel_index = task.new_ele[
                        task.new_ele["Label"] == region
                    ]["Channel"].tolist()
                    good_channel = np.setdiff1d(channel_index, task.bad_contact)
                else:
                    good_channel = np.arange(task.N_channels)
                    good_channel = np.setdiff1d(good_channel, task.bad_contact)
                if len(good_channel) < 2:
                    # delete the region from regions
                    regions_to_plot.remove(region)
                    continue
                for i in range(n_iter):
                    print(f"Computing the accuracy for {region} ...")
                    LSTM_trans_accuracies.append(
                        train_test_model(
                            task,
                            dPCA=True,
                            random_state=20240526 + i,
                            model="LSTM",
                            n_components=20,
                            band_type="high_gamma",
                            epoches=100,
                            region=region,
                        )
                    )

                    pure_chance.append(
                        train_test_model(
                            task,
                            dPCA=False,
                            random_state=20240526 + i,
                            model="Chance",
                            band_type="high_gamma",
                            region=region,
                        )
                    )

            # Rearrange the LSTM_trans_accuracies
            LSTM_trans_accuracies = np.reshape(
                LSTM_trans_accuracies, (len(regions_to_plot), n_iter)
            )
            pure_chance = np.reshape(
                pure_chance, (len(regions_to_plot), n_iter)
            )
            # calculate the mean of each location
            mean_pure_chance = np.mean(pure_chance, axis=1)
            # Create the boxplot
            fig, ax = plt.subplots()
            ax.boxplot(LSTM_trans_accuracies.T)
            # Plot the mean of pure chance
            ax.plot(
                np.arange(1, len(regions_to_plot) + 1),
                mean_pure_chance,
                "r--",
                label="Chance",
            )
            ax.set_ylim([0.3, 1])

            # Add titles and labels
            ax.set_title("Model Accuracy Comparison")
            ax.set_xlabel("Location")
            ax.set_ylabel("Accuracy")
            ax.set_xticklabels(regions_to_plot)
            # Save the plot
            save_location = os.path.join(
                task.output_dir, "Model_Accuracy_Comparison.png"
            )
            plt.savefig(save_location)
            # Show the plot
            plt.show()

        else:
            data_to_train = task.backupdata
            accuracy_score = task.train_test_dataset(
                data_to_train, dr_method="PCA", n_dimensions=30, model="LDA"
            )
            #
            band = ["alpha", "beta", "gamma", "high_gamma"]
            N_channels = task.N_channels

            print("Start computing the neuron dropping curve...")

            indicies_all = []
            result_all = []
            bad_contacts = []

            for fred_band in band:
                print(
                    "Computing the neuron dropping curve for"
                    f" {fred_band} band..."
                )

                dimensions = np.arange(1, 50, 5)

                result = Parallel(n_jobs=n_cpus)(
                    delayed(parallel_computing_accuracy_N_dimensions)(
                        task, n_dimensions=i, band=fred_band, model=model
                    )
                    for i in dimensions
                )
                # Get the first element of the result
                indices = [item[1] for item in result]
                result = [item[0] for item in result]
                bad_contact = [item[2] for item in result]

                indicies_all.append(indices)
                result_all.append(result)
                bad_contacts.append(bad_contact[0])

            max_accuracy_indices_freq = []
            max_accuracy_freq = []

            for freq_band in band:
                # Use the max accuracy index to get the index of the feature
                bandindex = {"alpha": 0, "beta": 1, "gamma": 2, "high_gamma": 3}
                # Get the index of the band
                ind = bandindex[freq_band]
                max_accuracy_indices = []
                max_accuracy = []
                for i in range(0, len(result_all[ind])):
                    max_accuracy_index = np.argmax(result_all[ind][i])
                    max_accuracy_index = np.array(
                        indicies_all[ind][i][max_accuracy_index]
                    )
                    max_accuracy_indices.append(max_accuracy_index)

                    key, value = list(task.re_organized_psd_range.items())[ind]
                    data = task.clustering({key: value})
                    # Delete bad contacts in bad_contacts
                    data = np.delete(data, bad_contacts[ind], axis=1)

                    # Select the data with max accuracy and last column
                    max_accuracy_index = np.append(max_accuracy_index, -1)
                    data = data[:, max_accuracy_index]

                    max_accuracy.append(task.train_test_dataset(data))

                max_accuracy_indices_freq.append(max_accuracy_indices)
                max_accuracy_freq.append(max_accuracy)

            # Save the max accuracy indices to json file
            task.max_accuracy_indices = max_accuracy_indices_freq
            task.save_to_json()

    print(taskID + " Complete")
