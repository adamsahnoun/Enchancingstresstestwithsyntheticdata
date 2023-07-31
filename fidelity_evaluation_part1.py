import pandas as pd

# Load the data
data = pd.read_csv('downGANtd.csv')

# Display the first few rows of the data
data.head()
# Check for missing values
missing_values = data.isnull().sum()

# Check data types
data_types = data.dtypes

# Summary statistics
summary_stats = data.describe()

# Unique sectors and symbols
unique_sectors = data['Sector'].nunique()
unique_symbols = data['Symbol'].nunique()

# Date range
date_range = data['Date'].min(), data['Date'].max()

missing_values, data_types, summary_stats, unique_sectors, unique_symbols, date_range
# Calculate the number of sequences
num_sequences = len(data) / 20

import matplotlib.pyplot as plt

# Select the first few sequences
first_sequences = data['Log Return'].values.reshape(-1, 20)[:5]

# Create a line plot for the first few sequences
plt.figure(figsize=(14, 7))
for i, sequence in enumerate(first_sequences):
    plt.plot(sequence, label=f'Sequence {i+1}')
plt.title('Log Return for the First 5 Sequences')
plt.xlabel('Day')
plt.ylabel('Log Return')
plt.legend()
plt.grid(True)
plt.show()

# Create a histogram of the log return values
plt.figure(figsize=(14, 7))
plt.hist(data['Log Return'], bins=30, alpha=0.7)
plt.title('Histogram of Log Return Values')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


import numpy as np

def autocorr(x):
    """Compute autocorrelation for a sequence."""
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

# Reshape the data into sequences
sequences = data['Log Return'].values.reshape(-1, 20)

# Compute autocorrelation for each sequence
autocorrelations = [autocorr(seq) for seq in sequences]

# Compute the average autocorrelation
average_autocorrelation = np.mean(autocorrelations, axis=0)

average_autocorrelation

# Load the synthetic data
synthetic_data = pd.read_csv('synth_down.csv')

# Display the first few rows of the synthetic data
synthetic_data.head()

# Check for missing values in the synthetic data
syn_missing_values = synthetic_data.isnull().sum()

# Check data types
syn_data_types = synthetic_data.dtypes

# Summary statistics
syn_summary_stats = synthetic_data.describe()

# Unique sectors and symbols
syn_unique_sectors = synthetic_data['Sector'].nunique()
syn_unique_symbols = synthetic_data['Symbol'].nunique()

# Date range
syn_date_range = synthetic_data['Date'].min(), synthetic_data['Date'].max()

# Calculate the number of sequences in the synthetic data
syn_num_sequences = len(synthetic_data) / 20

syn_missing_values, syn_data_types, syn_summary_stats, syn_unique_sectors, syn_unique_symbols, syn_date_range, syn_num_sequences

# Select the first few sequences of the synthetic data
syn_first_sequences = synthetic_data['Log Return'].values.reshape(-1, 20)[:5]

# Create a line plot for the first few sequences
plt.figure(figsize=(14, 7))
for i, sequence in enumerate(syn_first_sequences):
    plt.plot(sequence, label=f'Sequence {i+1}')
plt.title('Log Return for the First 5 Synthetic Sequences')
plt.xlabel('Day')
plt.ylabel('Log Return')
plt.legend()
plt.grid(True)
plt.show()

# Create a histogram of the log return values
plt.figure(figsize=(14, 7))
plt.hist(synthetic_data['Log Return'], bins=30, alpha=0.7)
plt.title('Histogram of Log Return Values for the Synthetic Data')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Compute autocorrelation for each sequence in the synthetic data
syn_sequences = synthetic_data['Log Return'].values.reshape(-1, 20)
syn_autocorrelations = [autocorr(seq) for seq in syn_sequences]

# Compute the average autocorrelation for the synthetic data
syn_average_autocorrelation = np.mean(syn_autocorrelations, axis=0)

syn_average_autocorrelation

summary_comparison = pd.concat([summary_stats, syn_summary_stats], axis=1)

# Drop the extra columns from the synthetic data's summary statistics
syn_summary_stats = syn_summary_stats.drop(columns=['Unnamed: 0', 'example_id'])

# Correct the column names for comparison
summary_comparison = pd.concat([summary_stats, syn_summary_stats], axis=1)
summary_comparison.columns = pd.MultiIndex.from_product([['Original', 'Synthetic'], summary_stats.columns])

# Display the comparison
summary_comparison

# Plot histograms of original and synthetic data on the same plot
plt.figure(figsize=(14, 7))
plt.hist(data['Log Return'], bins=30, alpha=0.7, label='Original', color='blue', edgecolor='black')
plt.hist(synthetic_data['Log Return'], bins=30, alpha=0.7, label='Synthetic', color='orange', edgecolor='black')
plt.title('Histogram of Log Return Values')
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Plot autocorrelations of original and synthetic data on the same plot
plt.figure(figsize=(14, 7))
plt.plot(average_autocorrelation, label='Original', color='blue')
plt.plot(syn_average_autocorrelation, label='Synthetic', color='orange')
plt.title('Average Autocorrelation of Log Return Values')
plt.xlabel('Lag (days)')
plt.ylabel('Autocorrelation')
plt.legend()
plt.grid(True)
plt.show()


# Randomly sample a subset of the synthetic data
syn_sample = synthetic_data.sample(n=len(data), random_state=42)

plt.figure(figsize=(10, 10))
plt.scatter(data['Log Return'], syn_sample['Log Return'], alpha=0.5)
plt.title('Scatter Plot of Log Return Values')
plt.xlabel('Original Data')
plt.ylabel('Synthetic Data')
plt.grid(True)
plt.show()