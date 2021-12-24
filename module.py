import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# ----------------------------------------------------------------------

# human_idx = '01'
# filename = f'./data/movement_data/NM00{human_idx}.tsv'
markers = {"01": 12}
# fs = 100
rows_num = 36000
axes = ["X", "Y", "Z"]
# human = 1

# First and last segments always were Silence - 60s
# 1 - Middle Silence - 60s
# 2 - Meditation - 60s
# 3 - Salsa - 60s
# 4 - EDM - 60s
# Total duration of recording 360s
stim_types = {
    0: "Silence 1",
    1: "Middle Silence",
    2: "Meditation",
    3: "Salsa",
    4: "EDM",
    5: "Silence 2",
}
parts_amount = 6


# ----------------------------------------------------------------------


def build_signal_fft(sig, t_seq, name, build_phase=False, sr=128, ts=1):
    rfft_sig = np.fft.rfft(sig)
    mag_spectrum = np.abs(rfft_sig)

    xf = np.fft.rfftfreq(sr * ts, 1 / sr)

    ft = np.fft.fft(sig)

    half_len = int(len(t_seq) / 2)

    if build_phase:
        fig, axs = plt.subplots(3, 1, figsize=([20, 16]))
    else:
        fig, axs = plt.subplots(2, 1, figsize=([20, 13]))

    plt.locator_params(axis="x", nbins=40)

    axs[0].set_title(f"Input signal - {name}")
    axs[0].plot(t_seq, sig, color="r", label="Filtered signal")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude (V)")
    axs[0].legend(bbox_to_anchor=[1, 1], ncol=2, title="Legend")
    axs[0].grid(True)

    axs[1].set_title(f"Amplitude spectrum - {name}")
    axs[1].stem(xf, mag_spectrum, markerfmt=None)
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Amplitude ")
    axs[1].legend(bbox_to_anchor=[1, 1], ncol=2, title="Legend")
    axs[1].grid(True)

    if build_phase:
        phase = np.angle(ft)

        axs[2].set_title(f"Phase spectrum - {name}")
        axs[2].stem(t_seq, phase)
        axs[2].set_xlabel("Frequency (Hz)")
        axs[2].set_ylabel("Phase ")
        axs[2].legend(bbox_to_anchor=[1, 1], ncol=2, title="Legend")
        axs[2].grid(True)


# Dataframe with displacement data for 1 experiment
def create_displacement_experiment_data(df, exp_idx, fs):
    raw_data = {}

    # Separate different people and add time column
    for m in range(1, markers[exp_idx] + 1):
        time = np.linspace(0, rows_num / fs, rows_num)
        human_df = df[[f"S{m} X", f"S{m} Y", f"S{m} Z"]]
        human_df.columns = axes
        human_df.insert(0, "Time (s)", time)
        raw_data[m] = human_df

    # Calculate displacement (difference before two positions)
    for m in range(1, markers[exp_idx]):
        for position in axes:
            displ = np.roll(raw_data[m][position], -1) - raw_data[m][position]
            displ = displ.drop(displ.index[len(displ) - 1])
            raw_data[m].insert(0, f"d{position}", displ)

    return raw_data


def plot_data_axes(raw_data, human):
    fig, axs = plt.subplots(3, 1, figsize=(25, 15))
    axs[0].plot(raw_data[human]["Time (s)"], raw_data[human]["dX"], label="Sick EEG")
    axs[0].set_title("Amplitude data for X axis - Human 9")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Volume (mm)")
    axs[0].grid(True)

    axs[1].plot(raw_data[human]["Time (s)"], raw_data[human]["dY"], label="Sick EEG")
    axs[1].set_title("Amplitude data for Y axis - Human 9")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Volume (mm)")
    axs[1].grid(True)

    axs[2].plot(raw_data[human]["Time (s)"], raw_data[human]["dZ"], label="Sick EEG")
    axs[2].set_title("Amplitude data for Z axis - Human 9")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Volume (mm)")
    axs[2].grid(True)


def read_stimuli_in_dfs(raw_data, exp_idx, fs):
    stimuli_df = pd.read_csv("./data/sound_data/nm15_song_order.csv")
    stimuli_df.head()

    stimuled_humans = {}
    ticks_per_60_second = fs * 60

    # For each person in stage
    for person in range(1, markers[exp_idx] + 1):

        # Getting ready for stimules parts
        stimules = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None}

        # 0 and 5 are Silence
        stimuli_order = list(stimuli_df.iloc[int(exp_idx) - 1])
        stimuli_order[0] = 5
        stimuli_order.append(0)

        print(stimuli_order)

        # Fill in different parts of signals for each stimuli type / silence
        for s in range(len(stimuli_order)):
            part = raw_data[person].iloc[
                stimuli_order[s]
                * ticks_per_60_second : (stimuli_order[s] + 1)
                * ticks_per_60_second
            ]
            stimules[s] = part

        stimuled_humans[person] = stimules

    # Usage: stimuled_humans[1][1]
    return stimuled_humans


def build_six_parts_signal(stimuled_human, human_idx):
    fig, axs = plt.subplots(3, 1, figsize=(25, 15))
    sig_colors = ["#34495E", "#884EA0", "#3498DB", "#2ECC71", "#F4D03F", "#34495E"]
    for i in range(parts_amount):
        color = "#555015"
        axs[0].plot(
            stimuled_human[i]["Time (s)"], stimuled_human[i]["dX"], color=sig_colors[i]
        )
        axs[1].plot(
            stimuled_human[i]["Time (s)"], stimuled_human[i]["dY"], color=sig_colors[i]
        )
        axs[2].plot(
            stimuled_human[i]["Time (s)"], stimuled_human[i]["dZ"], color=sig_colors[i]
        )

    for j in range(len(axes)):
        axs[j].set_title(
            f"Amplitude for splitted {axes[j]} axis data - Human {human_idx}"
        )
        axs[j].set_xlabel("Time (s)")
        axs[j].set_ylabel("Volume (mm)")
        axs[j].grid(True)
