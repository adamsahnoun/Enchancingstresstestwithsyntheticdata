
# """
# Train the DGAN using Gretel Cloud API. Adapted from: https://github.com/gretelai/gretel-blueprints/blob/main/docs/notebooks/create_synthetic_data_from_dgan_api.ipynb
# 
# """

%%capture
!pip install gretel_client
# make imports
import math
 
from typing import List, Optional
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import yaml
 
from gretel_client import configure_session
from gretel_client.helpers import poll
from gretel_client.projects.projects import create_or_get_unique_project
from gretel_client.projects.models import read_model_config
 
from plotly.subplots import make_subplots
 
# Specify Gretel API Key
configure_session(api_key="prompt", cache="no", validate=True)
 
df = pd.read_csv('dGANtd') # for the crash model the dataset is downGANtd
 
# Setup config and train model
 
project = create_or_get_unique_project(name="DGAN-oil")

print(f"Follow model training at: {project.get_console_url()}")

# name the model
config = read_model_config("synthetics/time-series")

# Update the 'name' field
config["name"] = "time-series-dgan_recovery" # time-series-dgan_crash for the other GAN

# Update fields in the 'timeseries_dgan' model
model_config = config["models"][0]["timeseries_dgan"]



# Update fields in 'params'
params_config = model_config["params"]

# Number of time points to produce from 1 RNN cell in the generator. Must evenly divide max_sequence_len
params_config["sample_len"] = 1
# Length of training example
params_config["max_sequence_len"] = 20
# attribute generator noise dim
params_config["attribute_noise_dim"] = 10
# feature generator noise dim
params_config["feature_noise_dim"] = 32
# Number of hidden layers in the feed-forward MLP to create attributes in the GAN generator
params_config["attribute_num_layers"] = 3
# Number of units in each layer of the feed-forward MLP to create attributes in the GAN generator.
params_config["attribute_num_units"] = 100
# Number of LSTM layers
params_config["feature_num_layers"] = 1
# Number of LSTM layers in the RNN to create features in the GAN generator.
params_config["feature_num_units"] = 100
# Deactivate attribute discriminator - we only have 1 asset in the portfolio
params_config["use_attribute_discriminator"] = False
# setting the normalization to 1 means the model will generate values between -1 and 1 (tanh); 0 for sigmoid
params_config["normalization"] = 1
# Disable automatic scaling
params_config["apply_feature_scaling"] = False
params_config["apply_example_scaling"] = False
params_config["binary_encoder_cutoff"] = 150
params_config["generator_learning_rate"] = 0.00001
params_config["discriminator_learning_rate"] = 0.00001
params_config["attribute_discriminator_learning_rate"] = 0.00001
params_config["batch_size"] = 100
params_config["epochs"] = 5000


# Update other fields in 'timeseries_dgan'
model_config["attribute_columns"] = None
model_config["feature_columns"] = None
model_config["example_id_column"] = None
model_config["time_column"] = "auto"
model_config["discrete_columns"] = None
model_config["df_style"] = "long"

# Update 'generate'
model_config["generate"] = {
    "num_records": 5000,
    "max_invalid": None
}

config["notifications"] = None
config["label_predictors"] = None

model = project.create_model_obj(model_config=config, data_source=df)
model.submit_cloud()

poll(model)

# Get synthetic data
synthetic_df = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
synthetic_df.to_csv("synth_up.csv") # for the crash model it was saved as synth_down.csv