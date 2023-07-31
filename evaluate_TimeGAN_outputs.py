
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the synthetic data
synthetic_data = pd.read_csv('/mnt/data/synthetic_data_example.csv')

# Display the first few rows of the synthetic data
synthetic_data.head()


# Drop the unnecessary index column
synthetic_data = synthetic_data.drop(columns=["Unnamed: 0"])

# Plot the synthetic data sequences
plt.figure(figsize=(12, 6))
for i in range(len(synthetic_data)):
    plt.plot(synthetic_data.columns, synthetic_data.iloc[i], alpha=0.5)
plt.title('Synthetic Data Sequences')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.show()

# Load the real data
real_data = pd.read_csv('/mnt/data/real_data_example.csv')

# Display the first few rows of the real data
real_data.head()


# Flatten the synthetic and real data for KDE plot
synthetic_data_flat = synthetic_data.values.flatten()
real_data_flat = real_data.values.flatten()

# Create a DataFrame for easier seaborn plotting
data_df = pd.DataFrame({
    'Density': np.concatenate([synthetic_data_flat, real_data_flat]),
    'Type': np.concatenate([np.repeat('Synthetic', len(synthetic_data_flat)), np.repeat('Real', len(real_data_flat))])
})

# Create the KDE plot
plt.figure(figsize=(12, 6))
sns.kdeplot(data=data_df, x='Density', hue='Type')
plt.title('Density Plot of Real and Synthetic Data')
plt.xlabel('Value')
plt.show()

from statsmodels.tsa.stattools import acf

# Define a function to calculate autocorrelation at different lags
def calculate_autocorrelation(data, max_lag):
    autocorrelation = np.zeros(max_lag)
    for row in data.values:
        autocorrelation += acf(row, nlags=max_lag, fft=True)[1:]
    return autocorrelation / len(data)

# Calculate autocorrelation for real and synthetic data
max_lag = 10  # maximum lag to calculate autocorrelation for
autocorrelation_real = calculate_autocorrelation(real_data, max_lag)
autocorrelation_synthetic = calculate_autocorrelation(synthetic_data, max_lag)

# Plot the autocorrelation
plt.figure(figsize=(12, 6))
plt.plot(range(1, max_lag + 1), autocorrelation_real, marker='o', label='Real')
plt.plot(range(1, max_lag + 1), autocorrelation_synthetic, marker='o', label='Synthetic')
plt.title('Average Autocorrelation of Real and Synthetic Data')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.legend()
plt.show()