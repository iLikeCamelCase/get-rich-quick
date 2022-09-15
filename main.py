import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
)
from darts.metrics import mape, smape
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset

torch.manual_seed(1)
np.random.seed(1)
#import first 3rd of data and convert to panda dataframe
#pd_data_01 = pd.read_csv("datasets\SPY_sample_5min_04012022.csv", sep=",")
#import second 3rd of data and convert to panda dataframe
pd_data_02 = pd.read_csv("datasets\SPY_sample_5min_04042022.csv", sep=",")
#import thrid 3rd of data and convert to panda dataframe
pd_data_03 = pd.read_csv("datasets\SPY_sample_5min_04052022.csv", sep=",")

day_01 = TimeSeries.from_csv("datasets\SPY_sample_5min_04012022.csv", time_col="DateTime", value_cols=" Close", fill_missing_dates=True, freq="5T")
day_01.plot(new_plot=True)

plt.legend()
