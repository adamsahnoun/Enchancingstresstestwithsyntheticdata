
import matplotlib.pyplot as plt

# Import necessary libraries
import pandas as pd
import numpy as np

# Load the data
dotcom_bubble_data = pd.read_csv('/content/drive/MyDrive/dotcom_bubble.csv', index_col='Date', parse_dates=True)
gfc_data = pd.read_csv('/content/drive/MyDrive/gfc.csv', index_col='Date', parse_dates=True)

# Display the first few rows of each dataset
dotcom_bubble_data.head(), gfc_data.head()

# Calculate the daily returns
dotcom_bubble_data['Return'] = dotcom_bubble_data['Close'].pct_change()
gfc_data['Return'] = gfc_data['Close'].pct_change()

# Calculate and display the basic statistics
dotcom_stats = dotcom_bubble_data['Return'].describe()
gfc_stats = gfc_data['Return'].describe()

# Skewness and kurtosis
dotcom_stats['skewness'] = dotcom_bubble_data['Return'].skew()
dotcom_stats['kurtosis'] = dotcom_bubble_data['Return'].kurt()
gfc_stats['skewness'] = gfc_data['Return'].skew()
gfc_stats['kurtosis'] = gfc_data['Return'].kurt()

dotcom_stats, gfc_stats

# Function to calculate the drawdown and recovery period
def calculate_drawdown_and_recovery(data):
    # Calculate the cumulative return
    data['Cumulative Return'] = (1 + data['Return']).cumprod()

    # Calculate the running max
    data['Running Max'] = data['Cumulative Return'].cummax()

    # Calculate the drawdown
    data['Drawdown'] = data['Cumulative Return'] / data['Running Max'] - 1

    # Identify the bottom of the crisis
    bottom_of_crisis = data['Drawdown'].idxmin()

    # Calculate the recovery period
    recovery_period = (data.loc[bottom_of_crisis:, 'Cumulative Return'] / data.loc[bottom_of_crisis, 'Cumulative Return']).idxmax() - bottom_of_crisis

    return bottom_of_crisis, recovery_period

# Calculate the drawdown and recovery period for each crisis
dotcom_bottom, dotcom_recovery = calculate_drawdown_and_recovery(dotcom_bubble_data)
gfc_bottom, gfc_recovery = calculate_drawdown_and_recovery(gfc_data)

# Print the results
dotcom_bottom, dotcom_recovery, gfc_bottom, gfc_recovery

# Function to calculate the returns at the bottom and after recovery
def calculate_returns(data, bottom_of_crisis, recovery_period):
    # Calculate the return at the bottom of the crisis
    return_at_bottom = data.loc[bottom_of_crisis, 'Cumulative Return'] - 1

    # Calculate the return after the recovery period
    return_after_recovery = data.loc[bottom_of_crisis + recovery_period, 'Cumulative Return'] - 1

    return return_at_bottom, return_after_recovery

# Calculate the returns for each crisis
dotcom_return_at_bottom, dotcom_return_after_recovery = calculate_returns(dotcom_bubble_data, dotcom_bottom, dotcom_recovery)
gfc_return_at_bottom, gfc_return_after_recovery = calculate_returns(gfc_data, gfc_bottom, gfc_recovery)

# Print the results
dotcom_return_at_bottom, dotcom_return_after_recovery, gfc_return_at_bottom, gfc_return_after_recovery

# Load the synthetic data
synthetic_data = pd.read_csv('/content/drive/MyDrive/series_test1.csv')

# Drop the 'Date' column as it is not needed for the analysis
synthetic_data.drop('Date', axis=1, inplace=True)

# Function to calculate the maximum drawdown for a time series
def calculate_max_drawdown(series):
    # Calculate the daily returns
    series_return = series.pct_change()

    # Calculate the cumulative return
    cumulative_return = (1 + series_return).cumprod()

    # Calculate the running max
    running_max = cumulative_return.cummax()

    # Calculate the drawdown
    drawdown = cumulative_return / running_max - 1

    # Return the maximum drawdown
    return drawdown.min()

# Calculate the maximum drawdown for each synthetic time series
max_drawdowns = synthetic_data.apply(calculate_max_drawdown)

# Identify the series with the most extreme drawdown
most_extreme_series = max_drawdowns.idxmin()

# Calculate the volatility of the most extreme series
volatility_most_extreme = synthetic_data[most_extreme_series].pct_change().std()

# Calculate the volatilities of the Dotcom Bubble and the GFC
volatility_dotcom = dotcom_bubble_data['Return'].std()
volatility_gfc = gfc_data['Return'].std()

# Print the results
most_extreme_series, volatility_most_extreme, volatility_dotcom, volatility_gfc

# Calculate the maximum drawdown for each synthetic time series
max_drawdowns = synthetic_data.apply(calculate_max_drawdown)

# Sort the drawdowns and get the indices of the 10 most extreme ones
most_extreme_series = max_drawdowns.nsmallest(10)

# Create a dataframe for a nicer output
most_extreme_df = pd.DataFrame(most_extreme_series)
most_extreme_df.reset_index(inplace=True)
most_extreme_df.columns = ['Series', 'Max Drawdown']

# Calculate the volatility of the most extreme series
volatilities_most_extreme = [synthetic_data[series].pct_change().std() for series in most_extreme_df['Series']]

# Calculate the volatilities of the Dotcom Bubble and the GFC
volatility_dotcom = dotcom_bubble_data['Return'].std()
volatility_gfc = gfc_data['Return'].std()

# Add volatilities to the dataframe
most_extreme_df['Volatility'] = volatilities_most_extreme

# Print the results
print(most_extreme_df)
print("Volatility of Dotcom Bubble: ", volatility_dotcom)
print("Volatility of GFC: ", volatility_gfc)

# Function to calculate the mean return for a time series
def calculate_mean_return(series):
    return series.pct_change().mean()

# Function to calculate the volatility for a time series
def calculate_volatility(series):
    return series.pct_change().std()

# Calculate the mean return for each synthetic time series
mean_returns_synthetic = synthetic_data.apply(calculate_mean_return)

# Calculate the volatility for each synthetic time series
volatilities_synthetic = synthetic_data.apply(calculate_volatility)

# Calculate the average mean return and average volatility for the synthetic dataset
average_mean_return_synthetic = mean_returns_synthetic.mean()
average_volatility_synthetic = volatilities_synthetic.mean()

# Calculate the mean return for the original crises
mean_return_dotcom = dotcom_bubble_data['Return'].mean()
mean_return_gfc = gfc_data['Return'].mean()

# Print the results
average_mean_return_synthetic, average_volatility_synthetic, mean_return_dotcom, mean_return_gfc, volatility_dotcom, volatility_gfc

"""The general attributes of the synthetic dataset and the two historical crises are as follows:

Synthetic Dataset:

The average mean return across all synthetic time series is approximately 0.0086%.
The average volatility across all synthetic time series is approximately 1.19%.
Dotcom Bubble:

The mean return was approximately 0.0056%.
The volatility was approximately 1.09%.
Global Financial Crisis (GFC):

The mean return was approximately 0.013%.
The volatility was approximately 1.61%.
Comparing these metrics, we can see that the synthetic dataset's average mean return is higher than that of the Dotcom Bubble, but lower than that of the GFC. The average volatility of the synthetic dataset is higher than that of the Dotcom Bubble and lower than that of the GFC.

This suggests that the synthetic dataset, on average, exhibits characteristics somewhere between those of the Dotcom Bubble and the GFC. However, there is considerable variation within the synthetic dataset
"""

# Function to calculate the drawdown for a time series
def calculate_drawdown(series):
    # Calculate the daily returns
    series_return = series.pct_change()

    # Calculate the cumulative return
    cumulative_return = (1 + series_return).cumprod()

    # Calculate the running max
    running_max = cumulative_return.cummax()

    # Calculate the drawdown
    drawdown = cumulative_return / running_max - 1

    # Return the drawdown series
    return drawdown

# Calculate the drawdown for the original crises
dotcom_bubble_data['Drawdown'] = calculate_drawdown(dotcom_bubble_data['Close'])
gfc_data['Drawdown'] = calculate_drawdown(gfc_data['Close'])

# Plot the distribution of maximum drawdowns
plt.figure(figsize=(12, 6))
plt.hist(max_drawdowns, bins=30, alpha=0.75, label='Synthetic Data')
plt.axvline(dotcom_bubble_data['Drawdown'].min(), color='r', linestyle='dashed', linewidth=2, label='Dotcom Bubble')
plt.axvline(gfc_data['Drawdown'].min(), color='g', linestyle='dashed', linewidth=2, label='GFC')
plt.title('Distribution of Maximum Drawdowns')
plt.xlabel('Maximum Drawdown')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot the distribution of volatilities
plt.figure(figsize=(12, 6))
plt.hist(volatilities_synthetic, bins=30, alpha=0.75, label='Synthetic Data')
plt.axvline(volatility_dotcom, color='r', linestyle='dashed', linewidth=2, label='Dotcom Bubble')
plt.axvline(volatility_gfc, color='g', linestyle='dashed', linewidth=2, label='GFC')
plt.title('Distribution of Volatilities')
plt.xlabel('Volatility')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Function to calculate the percentage loss from the start of the crisis to the lowest point
def calculate_loss_to_lowest_point(series):
    # Calculate the daily returns
    series_return = series.pct_change()

    # Calculate the cumulative return
    cumulative_return = (1 + series_return).cumprod()

    # Calculate the percentage loss to the lowest point
    loss_to_lowest_point = cumulative_return.min() - 1

    # Return the percentage loss to the lowest point
    return loss_to_lowest_point

# Calculate the percentage loss to the lowest point for each synthetic time series
losses_to_lowest_point = synthetic_data.apply(calculate_loss_to_lowest_point)

# Calculate the percentage loss to the lowest point for the original crises
loss_to_lowest_point_dotcom = dotcom_bubble_data['Cumulative Return'].min() - 1
loss_to_lowest_point_gfc = gfc_data['Cumulative Return'].min() - 1

# Recalculate the cumulative returns for the original crises
dotcom_bubble_data['Cumulative Return'] = (1 + dotcom_bubble_data['Return']).cumprod() - 1
gfc_data['Cumulative Return'] = (1 + gfc_data['Return']).cumprod() - 1

# Calculate the percentage loss to the lowest point for the original crises
loss_to_lowest_point_dotcom = dotcom_bubble_data['Cumulative Return'].min()
loss_to_lowest_point_gfc = gfc_data['Cumulative Return'].min()

# Plot the distribution of percentage losses to the lowest point
plt.figure(figsize=(12, 6))
plt.hist(losses_to_lowest_point, bins=30, alpha=0.75, label='Synthetic Data')
plt.axvline(loss_to_lowest_point_dotcom, color='r', linestyle='dashed', linewidth=2, label='Dotcom Bubble')
plt.axvline(loss_to_lowest_point_gfc, color='g', linestyle='dashed', linewidth=2, label='GFC')
plt.title('Distribution of Percentage Losses to the Lowest Point')
plt.xlabel('Percentage Loss to Lowest Point')
plt.ylabel('Frequency')
plt.legend()
plt.show()

def plot_data(real_data1, real_data2, synthetic_data, selected_cols):
    # Ensure that the length of the plotted series is the same as the synthetic series
    real_data1 = real_data1.iloc[:len(synthetic_data)]
    real_data2 = real_data2.iloc[:len(synthetic_data)]

    # Normalize the data so they can be visualized on the same scale
    real_data1 = (real_data1 - real_data1.min()) / (real_data1.max() - real_data1.min())
    real_data2 = (real_data2 - real_data2.min()) / (real_data2.max() - real_data2.min())
    synthetic_data = synthetic_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    plt.figure(figsize=(12, 6))
    plt.plot(real_data1.values, label='GFC')
    plt.plot(real_data2.values, label='Dotcom')

    for col in selected_cols:
        plt.plot(synthetic_data[col].values, label=f'Synthetic Data {col}')

    plt.title('Time Series Comparison')
    plt.xlabel('Trading Days')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.show()

selected_cols = ['84']  # replace with your selected columns
plot_data(gfc_data['Close'], dotcom_bubble_data['Close'], synthetic_data, selected_cols)

"""Series  Max Drawdown  Volatility
     73     -0.628816    0.013288
     57     -0.609065    0.012360
    142     -0.600006    0.013163
     23     -0.594881    0.012674
    169     -0.593337    0.012881
    193     -0.561499    0.014446
     61     -0.537170    0.011913
    156     -0.530361    0.011539
     84     -0.522205    0.013131
     66     -0.520026    0.011137
Volatility of Dotcom Bubble:  0.010921987344067378
Volatility of GFC:  0.016129695833734622
"""

