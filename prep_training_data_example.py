# run pip install ydata_synthetic and restart runtime afterwards (needed for the processing function)

# Import packages
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from ydata_synthetic.preprocessing.timeseries import processed_stock

# Define simple functions for data handling
def trim_to_60_days(series_list, phase):
    trimmed_series_list = []
    for series in series_list:
        if len(series) > 60:
            if phase == 'down':
                trimmed_series = series[-60:]  # Keep the last 60 days
            elif phase == 'up':
                trimmed_series = series[:60]  # Keep the first 60 days
            trimmed_series_list.append(trimmed_series)
        else:
            trimmed_series_list.append(series)
    return trimmed_series_list

def series_to_df(series):
    df = pd.DataFrame(series)
    df.columns = ['Close']
    return df

def arrays_to_dataframe(array_list):
    # Converting each array into a list, and creating a dataframe
    array_list = np.squeeze(array_list)
    df = pd.DataFrame(array_list)

    return df

# Define the tickers
tickers = ['^GSPC']

# Dotcom
start_date1 = '2000-3-24'
end_date1 = '2007-5-30'
# GFC
start_date2 = '2007-10-9'
end_date2 = '2013-3-28'
# China
start_date3 = '2015-8-1'
end_date3 = '2016-07-28'
# Covid
start_date4 = '2019-02-19'
end_date4 = '2023-02-19'

# Pull the data
data1 = yf.download(tickers, start=start_date1, end=end_date1)
bottom_date1 = data1['Close'].idxmin()
data2 = yf.download(tickers, start=start_date2, end=end_date2)
bottom_date2 = data2['Close'].idxmin()
data3 = yf.download(tickers, start=start_date3, end=end_date3)
bottom_date3 = data3['Close'].idxmin()
data4 = yf.download(tickers, start=start_date4, end=end_date4)
bottom_date4 = data4['Close'].idxmin()
# Split the time series
N = 90  # For example
M = 86 # For example

down_phase1 = data1['Close'].loc[(bottom_date1 - pd.DateOffset(days=N)):bottom_date1]
up_phase1 = data1['Close'].loc[bottom_date1:(bottom_date1 + pd.DateOffset(days=M))]
down_phase2 = data2['Close'].loc[(bottom_date2 - pd.DateOffset(days=N)):bottom_date2]
up_phase2 = data2['Close'].loc[bottom_date2:(bottom_date2 + pd.DateOffset(days=M))]
down_phase3 = data3['Close'].loc[(bottom_date3 - pd.DateOffset(days=N)):bottom_date3]
up_phase3 = data3['Close'].loc[bottom_date3:(bottom_date3 + pd.DateOffset(days=M))]
down_phase4 = data4['Close'].loc[(bottom_date4 - pd.DateOffset(days=N)):bottom_date4]
up_phase4 = data4['Close'].loc[bottom_date4:(bottom_date4 + pd.DateOffset(days=M))]

# Concatenate all down phases and up phases to 1 time series
# This should be done manually for all phases for each crisis
down_phases = [down_phase1, down_phase2, down_phase3, down_phase4]  # Add more down phases here
up_phases = [up_phase1, up_phase2, up_phase3, up_phase4]  # Add more up phases here

# Order the phases
down_phases = sorted(down_phases, key=lambda x: x[-1], reverse=True)  # Order by last value
up_phases = sorted(up_phases, key=lambda x: x[-1])  # Order by last value

# Concatenate the phases
down_phases = trim_to_60_days(down_phases, 'down')
up_phases = trim_to_60_days(up_phases, 'up')
# store to dataframes
down1 = series_to_df(down_phases[0])
down2 = series_to_df(down_phases[1])
down3 = series_to_df(down_phases[2])
down4 = series_to_df(down_phases[3])
# apply the processing function (processed_stock.py for closer inspection, from the original TimeGAN implementation)
rd1 = processed_stock(path='df1.csv', seq_len=20)
rd2 = processed_stock(path='df2.csv', seq_len=20)
rd3 = processed_stock(path='df3.csv', seq_len=20)
rd4 = processed_stock(path='df4.csv', seq_len=20)
# Create list of arrays. Each array is a training example
stock_data = []
rds = [rd1, rd2, rd3, rd4]
for i in rds:
  for j in i:
    stock_data.append(j)
# Store the list into a dataframe
real_data = arrays_to_dataframe(stock_data)
# Save to .csv file <-------- Training data ready!
real_data.to_csv('/content/drive/MyDrive/real_data_example.csv')