import pandas as pd
import csv
import codecs
import math
import numpy as np
from scipy import signal, stats
import os

header_names = ["Sig1", "Sig2", "Sig3", "Sig4", "Sig5"]
sample_mass_line = "Size"
data_start_line = "StartOfData"
temp_heading = "Temperature (°C)"
cp_heading = "Heat Capacity (mJ/°C)"
time_heading = "Time (min)"


class DSCDataFrame():
    def __init__(self, name):
        self.sample_mass = float(0)
        self.name = os.path.basename(name).split(".")[0]

    def create_data_frame(self, data, headers):
        self.data_frame = pd.DataFrame(data, columns=headers, dtype=float)


def import_dsc_data(file, verbose=False):
    data = DSCDataFrame(file)
    data_header_names = []
    data_list = []
    with codecs.open(file, "rU", "utf-16") as data_file:
        reader = csv.reader(data_file, delimiter="\t")
        header_row = next(reader)
        if verbose:
            print("Loading headers from data file...")
        while (header_row[0] != data_start_line):
            if header_row[0] == sample_mass_line:
                data.sample_mass = float(header_row[1])
            elif header_row[0] in header_names:
                index = header_names.index(header_row[0])
                data_header_names.append(header_row[1])
            header_row = next(reader)

        if verbose:
            print("Loading data from data file...")
        data_row = next(reader)
        while (data_row is not None):
            data_dict = {}
            for i, point in enumerate(data_row):
                data_dict[data_header_names[i]] = point
                data_list.append(data_dict)
            try:
                data_row = next(reader)
            except StopIteration:
                if verbose:
                    print("Finished loading data from data file...")
                break
        data.create_data_frame(data_list, data_header_names)
    return data


def select_between_range(dsc_df, start, end, col, verbose=False):
    if verbose:
        print("Selecting data from " + str(start) + " to " + str(end) + " from column [" + col + "]...")
    d = dsc_df[(dsc_df[col] > start) & (dsc_df[col] < end)]
    return d


def compute_gaussian(X, t_g, width, stp, magic_number, verbose=False):
    if verbose:
        print("Computing Gaussian with following parameters...")
        print("%5.2f, %5.2f, %5.2f, %5.2f" % (t_g, width, stp, magic_number))
    base = (stp / (magic_number * width))
    dif = X - t_g
    power = (dif / width).pow(2)
    return (base * np.exp((-1) * power))


def compute_cumulative_guassuan(gaussian, col, verbose=False):
    if verbose:
        print("Computing cumalative gaussian...")
    data_list = []
    sum = 0

    for value in gaussian:
        data_dict = {}
        sum += value
        data_dict[col] = sum
        data_list.append(data_dict)
    return pd.DataFrame(data_list)


def inverse_cumulative_gaussian(gaussian, col, verbose=False):
    if verbose:
        print("Computing inverse cumulative gaussian...")
    gaussian = np.array(gaussian)
    data_list = []

    for i in range(0, len(gaussian)):
        data_dict = {}
        sum = np.sum(gaussian[i:-1])
        data_dict[col] = sum
        data_list.append(data_dict)
    return pd.DataFrame(data_list)


def compute_enthaply_distro(X, enthalpy, width, max, verbose=False):
    if verbose:
        print("Computing enthalpy distribution 1 with following parameters...")
        print("%5.2f, %5.2f, %5.2f" % (enthalpy, width, max))
    em = math.sqrt(-math.log(.5))
    base = (em * (X - enthalpy)) / width
    return max * np.exp((-1) * (base).pow(2))


def compute_enthalpy_disro_2(X, enthalpy, width, max, verbose=False):
    if verbose:
        print("Computing enthalpy distribution 2 with following parameters...")
        print("%5.2f, %5.2f, %5.2f" % (enthalpy, width, max))
    return max / (1 + ((X - enthalpy) / width).pow(2))


def interp_temp_cp(df, start, step_size, max_step, verbose=False):
    if verbose:
        print("Interpolating data based on following parameters...")
        print("%d, %5.2f, %d" % (start, step_size, max_step))
    step = start
    data_list = []

    for i in range(0, max_step):
        data_dict = {}

        most_close_index = (df[temp_heading] - step).abs().idxmin()

        if most_close_index == df[temp_heading].first_valid_index():
            temp_1 = df.at[most_close_index, temp_heading]
            cp_1 = df.at[most_close_index, cp_heading]
        else:
            temp_1 = df.at[most_close_index - 1, temp_heading]
            cp_1 = df.at[most_close_index - 1, cp_heading]
        if most_close_index == df[temp_heading].last_valid_index():
            temp_2 = df.at[most_close_index, temp_heading]
            cp_2 = df.at[most_close_index, cp_heading]
        else:
            temp_2 = df.at[most_close_index + 1, temp_heading]
            cp_2 = df.at[most_close_index + 1, cp_heading]

        intrep_cp = linear_interp(step, temp_1, temp_2, cp_1, cp_2)
        data_dict[cp_heading] = intrep_cp
        data_dict[temp_heading] = step

        step += step_size
        data_list.append(data_dict)
    return pd.DataFrame(data_list)


def model_combonation(enthalpy, temp_range, enthalpy_distro, enthalpy_distro_2, tg_model, ratio, verbose=False):
    if verbose:
        print("Adding distribution models...")
    full_model = np.zeros(len(temp_range))
    scale_1 = 1 + ratio
    scale_2 = 1 - ratio
    for i, temperature in enumerate(temp_range):
        if temperature < enthalpy:
            full_model[i] = tg_model.at[i] + np.array(enthalpy_distro_2)[i] * scale_2
        else:
            full_model[i] = tg_model.at[i] + np.array(enthalpy_distro)[i] * scale_1
    return full_model


def linear_interp(x_1, x_2, x_3, y_1, y_2):
    return y_1 + (y_2 - y_1) * (x_1 - x_2) / (x_3 - x_2)


def scale(X):
    return np.divide(X, np.absolute(X).max())


def bin_first_deriv(X, smooth_interations=10, to_zero_tol=.005, parition_size=10):
    smoothed = signal.savgol_filter(X, 9, 4, 0)
    first = signal.savgol_filter(smoothed, 9, 4, 1)
    for i in range(0, smooth_interations):
        first = signal.savgol_filter(first, 9, 4, 0)
    l = (len(first))
    trim_length = l - (l % parition_size)
    f = first
    f[np.absolute(f) < to_zero_tol] = 0
    number_of_spilts = trim_length / parition_size
    parts = np.split(f[0:trim_length], number_of_spilts)

    signs = np.zeros(len(parts))
    for i in range(0, len(parts)):
        mean = np.mean(parts[i])
        mode = stats.mode(parts[i])[0]
        if mode == 0:
            signs[i] = 0
        else:
            if mean < 0:
                signs[i] = -1
            else:
                signs[i] = 1

    scaled = np.zeros(len(f))
    last_change_index = 0
    changes = []
    for i in range(0, len(signs)):
        if i != 0:
            if signs[i] != signs[i - 1]:
                if i == (len(signs) - 1):
                    changes.append((int(signs[i]), (0, i * parition_size)))
                elif len(changes) == 0:
                    changes.append((int(signs[i - 1]), (0, i * parition_size)))
                else:
                    changes.append((int(signs[i - 1]), (last_change_index, i * parition_size)))
                last_change_index = i * parition_size

        scaled[i * parition_size: (i + 1) * parition_size] = signs[i]

    return (scaled, changes)


def bin_pos_neg(X, to_zero_tol=.005, parition_size=10):
    l = (len(X))
    trim_length = l - (l % parition_size)
    f = X
    f[np.absolute(f) < to_zero_tol] = 0
    number_of_spilts = l / parition_size
    parts = np.split(f[0:trim_length], number_of_spilts)

    signs = np.zeros(len(parts))
    for i in range(0, len(parts)):
        mean = np.mean(parts[i])
        mode = stats.mode(parts[i])[0]
        if mode == 0:
            signs[i] = 0
        else:
            if mean < 0:
                signs[i] = -1
            else:
                signs[i] = 1

    scaled = np.zeros(len(f))
    last_change_index = 0
    changes = []
    for i in range(0, len(signs)):
        if i != 0:
            if signs[i] != signs[i - 1]:
                if i == (len(signs) - 1):
                    changes.append((int(signs[i]), (0, i * parition_size)))
                elif len(changes) == 0:
                    changes.append((int(signs[i - 1]), (0, i * parition_size)))
                else:
                    changes.append((int(signs[i - 1]), (last_change_index, i * parition_size)))
                last_change_index = i * parition_size

        scaled[i * parition_size: (i + 1) * parition_size] = signs[i]

    return (scaled, changes)


def suggest_overall_interest_regions(change_list):
    zero_regions = []

    for pair in change_list:
        if pair[0] == 0:
            zero_regions.append(pair[1])
    if len(zero_regions) == 1:
        return zero_regions[0]
    if len(zero_regions) > 1:
        top_two = sorted(zero_regions, reverse=True, key=lambda region: region[1] - region[0])[0:2]
        return (top_two[0][0], top_two[1][1])
    else:
        return (change_list[0][1][0], change_list[-1][1][1])


def suggest_linear_region(change_list):
    zero_regions = []
    for pair in change_list:
        if pair[0] == 0:
            zero_regions.append(pair[1])
    return sorted(zero_regions, reverse=True, key=lambda region: region[1] - region[0])[0]


def suggest_tg_region(change_list):
    for i, pair in enumerate(change_list):
        if i != len(change_list):
            # print(pair[0])
            # print(change_list[i+1][0])
            try:
                if pair[0] == 1 and change_list[i + 1][0] == -1:
                    return (pair[1][0], change_list[i + 2][1][1])
            except:
                return change_list[0][1]
    return change_list[0][1]
