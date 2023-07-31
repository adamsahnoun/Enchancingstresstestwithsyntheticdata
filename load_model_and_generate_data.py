# Run !pip install ydata_synthetic if not already and restart runtime afterwards

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN
import pandas as pd
import numpy as np

def arrays_to_dataframe(array_list):
    # Converting each array into a list, and creating a dataframe
    array_list = np.squeeze(array_list)
    df = pd.DataFrame(array_list)

    return df

synth = TimeGAN.load('synthesizer_down.pkl')
synth_data = synth.sample(160)
synth_data = arrays_to_dataframe(synth_data)
synth_data.to_csv('synthetic_data_example.csv')