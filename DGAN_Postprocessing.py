"""
POSTPROCESSING TO CREATE HYPOTHETICAL SCENARIOS - multiple batches (at least 2 of 5000 * 20day sequences) of synthetic data needed for around 200 full scenarios
"""

import datetime
#imports
from psutil import virtual_memory
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from os import path
import os

def generate_combined_series_no_norm(crash_arrays, recovery_arrays, first_value):
    # Concatenate the crash and recovery arrays
    crashes_concat = np.concatenate(crash_arrays)
    recoveries_concat = np.concatenate(recovery_arrays)

    # Calculate the relative changes
    c_changes = np.exp(crashes_concat)
    r_changes = np.exp(recoveries_concat)

    # Sanity check
    if c_changes.prod() > 1:
      return None
    if r_changes.prod() < 1:
      return None
    # Concatenate all arrays in the combined list
    concat_array = np.concatenate([c_changes, r_changes])

    # Cumulatively apply changes
    result = np.cumprod(concat_array) * first_value

    return result


def sample_arrays(crash_arrays: List[np.ndarray], recovery_arrays: List[np.ndarray], start_value: int) -> List[np.ndarray]:
    len_crash = len(crash_arrays)  # get the length of crash arrays
    len_recovery = len(recovery_arrays)  # get the length of recovery arrays

    crash_start_index = 0
    recovery_start_index = 0

    combined_data = []

    while crash_start_index < len_crash and recovery_start_index < len_recovery:
        rando = np.random.randint(5, 21)  # generating a random integer between 5 and 20
        crash_end_index = crash_start_index + rando
        recovery_end_index = recovery_start_index + (38 - rando)

        # check if samples would be empty, if so break the loop
        if crash_end_index > len_crash or recovery_end_index > len_recovery:
            break

        # taking samples from each list
        crash_sample = crash_arrays[crash_start_index:crash_end_index]
        recovery_sample = recovery_arrays[recovery_start_index:recovery_end_index]

        # combining samples
        combined_sample = generate_combined_series_no_norm(crash_sample, recovery_sample, start_value)

        # check if combined sample is valid
        if combined_sample is not None:
            combined_data.append(combined_sample)

        # update start indexes for next iteration
        crash_start_index = crash_end_index
        recovery_start_index = recovery_end_index

    return combined_data

def dataframe_to_arrays(df):
    # Split 'Log Return' column into arrays of length 20
    arrays = np.array_split(df['Log Return'].to_numpy(), len(df) // 20)

    return arrays

syn = pd.read_csv('syn_up.csv')
syn_down = pd.read_csv("syn_down.csv")
recovery_arrays_sample = dataframe_to_arrays(syn)
crash_arrays_sample = dataframe_to_arrays(syn_down)


#scale with initial value
a = sample_arrays(crash_arrays_sample, recovery_arrays_sample, 4400)

from typing import List

def arrays_to_df(arrays: List[np.ndarray]) -> pd.DataFrame:
    # Create a date range starting from July 1, 2023, with a length of 760 days
    date_range = pd.date_range(start='2023-07-01', periods=760)

    # Convert list of arrays into a DataFrame
    df = pd.DataFrame(np.column_stack(arrays), columns=[str(i) for i in range(len(arrays))])

    # Insert the date range as the first column
    df.insert(0, 'Date', date_range)

    return df

df = arrays_to_df(a)

file_path = "/content/drive/My Drive/series_test1.csv"
df.to_csv(file_path, index=True)