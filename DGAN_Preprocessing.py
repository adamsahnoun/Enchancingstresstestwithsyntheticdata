
import datetime
#imports
from psutil import virtual_memory
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from os import path
import os

#preprocessing
def compute_factor_returns(prices):
    """
    Compute factor returns from an array of prices
    """
    # Use np.diff to calculate the differences between subsequent elements
    # then divide these by the original elements (excluding the last one)
    return np.divide(np.diff(prices), prices[:-1])

def compute_log_returns(factor_returns):
    """
    Compute log returns from an array of factor returns
    """
    # Use np.log to calculate the natural logarithm of the factor returns
    return np.log(1 + factor_returns)

def split_data_by_crisis(ticker, start_date, end_date):
    """
    Downloads the data from Yahoo Finance, extracts close prices,
    identifies the crisis bottom and splits the data into two phases.
    """

    # a) Download daily data from Yahoo Finance
    # Data from Yahoo Finance is in chronological order (oldest first, newest last)
    data = yf.download(ticker, start=start_date, end=end_date)

    # b) Extract the close prices
    close_prices = data['Close'].values

    # c) Find the bottom of the crisis
    # argmin() returns the index of the minimum value, which corresponds to the crisis bottom
    crisis_bottom_index = np.argmin(close_prices)

    # d) Split the data into the down phase and recovery phase
    down_phase = close_prices[:crisis_bottom_index+1]  # +1 to include the bottom day in the down phase
    recovery_phase = close_prices[crisis_bottom_index:]

    return down_phase, recovery_phase

def plot_data(down_phase, recovery_phase):
    """
    Plots the down phase in red, recovery phase in green and the bottom of the crisis as a blue dot.
    """
    # Calculate the total number of days for x-axis
    total_days = len(down_phase) + len(recovery_phase)

    # Create an array representing each day
    days = np.array(range(total_days))

    # Plot the down phase in red
    plt.plot(days[:len(down_phase)], down_phase, 'r')

    # Plot the recovery phase in green
    plt.plot(days[len(down_phase):], recovery_phase, 'g')

    # Plot the bottom of the crisis as a blue dot
    plt.plot(len(down_phase)-1, down_phase[-1], 'bo')  # -1 because we want the last element of the down_phase

    # Add title and labels
    plt.title('Market Down Phase and Recovery Phase')
    plt.xlabel('Days')
    plt.ylabel('Price')

    plt.show()

def rdl(df, seq_len):
    """Load and preprocess real-world datasets.
    Args:
    - df: the Dataset
    - seq_len: sequence length for the TimeGAN
    Returns:
    - data: preprocessed data.
    """
    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(df) - seq_len):
        _x = df[i:i + seq_len].values  # convert DataFrame to ndarray
        temp_data.append(_x)
    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return data

down_phase1, recovery_phase1 = split_data_by_crisis('^GSPC', '2000-03-24', '2007-05-30')
down_phase2, recovery_phase2 = split_data_by_crisis('^GSPC', '2007-10-09', '2013-03-28')

# down phase 1
fr_down1 = compute_factor_returns(down_phase1)
lr_down1 = compute_log_returns(fr_down1)
# down phase 2
fr_down2 = compute_factor_returns(down_phase2)
lr_down2 = compute_log_returns(fr_down2)
# up phase 1
fr_up1 = compute_factor_returns(recovery_phase1)
lr_up1 = compute_log_returns(fr_up1)
# up phase 2
fr_up2 = compute_factor_returns(recovery_phase2)
lr_up2 = compute_log_returns(fr_up2)

# make dataframes
dwn1 = series_to_df(lr_down1)
dwn2 = series_to_df(lr_down2)
up1 = series_to_df(lr_up1)
up2 = series_to_df(lr_up2)

# apply the modified processing function
training_down1 = rdl(dwn1,20)
print("Down1 :",len(training_down1), training_down1[0].shape, type(training_down1[0]))
training_down2 = rdl(dwn2,20)
print("Down2 :",len(training_down2), training_down2[0].shape, type(training_down2[0]))

down_phases_training_data = []
for i in [training_down1, training_down2]:
  for j in i:
    down_phases_training_data.append(j)

training_up1 = rdl(up1,20)
print("Up1 :",len(training_up1), training_up1[0].shape, type(training_up1[0]))
training_up2 = rdl(up2,20)
print("Up2 :",len(training_up2), training_up2[0].shape, type(training_up2[0]))

up_phases_training_data = []
for i in [training_up1, training_up2]:
  for j in i:
    up_phases_training_data.append(j)

def arrays_to_dataframe(arrays): # create the dataframe expected by the DoppelGANger
    # Generate dates
    start_date = datetime(2022, 6, 1)
    dates = [(start_date + timedelta(days=i%20)).strftime('%Y-%m-%d') for i in range(len(arrays)*20)]

    # Flatten arrays and create DataFrame
    flattened_arrays = np.concatenate(arrays).flatten()
    df = pd.DataFrame({
        'Date': dates,
        'Sector': [1]*len(flattened_arrays),
        'Symbol': ['^GSPC']*len(flattened_arrays),
        'Log Return': flattened_arrays
    })

    return df

def dataframe_to_arrays(df):
    # Split 'Log Return' column into arrays of length 20
    arrays = np.array_split(df['Log Return'].to_numpy(), len(df) // 20)

    return arrays

# store
df = arrays_to_dataframe(up_phases_training_data)
df.to_csv('dGANtd.csv', index=False)
df = arrays_to_dataframe(down_phases_training_data)
df.to_csv('downGANtd.csv', index=False)