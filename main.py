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

from darts.metrics import mape, smape
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from sklearn.preprocessing import MinMaxScaler
from darts.utils.missing_values import missing_values_ratio
from darts.datasets import AusBeerDataset
from darts.dataprocessing.transformers.missing_values_filler import MissingValuesFiller

# global timeseries models
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
)


torch.manual_seed(1)
np.random.seed(1)
#import first 3rd of data and convert to panda dataframe
#pd_data_01 = pd.read_csv("datasets\SPY_sample_5min_04012022.csv", sep=",")
#import second 3rd of data and convert to panda dataframe
pd_data_02 = pd.read_csv("datasets\SPY_sample_5min_04042022.csv", sep=",")
#import thrid 3rd of data and convert to panda dataframe
pd_data_03 = pd.read_csv("datasets\SPY_sample_5min_04052022.csv", sep=",")

day_01 = TimeSeries.from_csv("datasets\SPY_sample_5min_04012022.csv", time_col="DateTime", value_cols=" Close", fill_missing_dates=True, freq="5T")
#day_01 = TimeSeries.from_csv("SPY_5min_sample.csv", time_col="DateTime", value_cols=" Close", fill_missing_dates=True, freq="5T")
day_02 = TimeSeries.from_csv("datasets\SPY_sample_5min_04042022.csv", time_col="DateTime", value_cols=" Close", fill_missing_dates=True, freq="5T")
day_03 = TimeSeries.from_csv("datasets\SPY_sample_5min_04052022.csv", time_col="DateTime", value_cols=" Close", fill_missing_dates=True, freq="5T")
val = TimeSeries.from_csv("datasets\QQQ_firstratedatacom1.csv", time_col="DateTime", value_cols=" Close", fill_missing_dates=True, freq="1T")
"""
day_01.plot(new_plot=True)
day_02.plot()
day_03.plot()

scaler_01, scaler_02, scaler_03 = Scaler(), Scaler(), Scaler()
series_01_scaled = scaler_01.fit_transform(scaler_01)
series_02_scaled = scaler_02.fit_transform(scaler_02)
series_03_scaled = scaler_03.fit_transform(scaler_03)
"""



transformer = MissingValuesFiller()
series_01_scaled = transformer.transform(day_01)
series_02_scaled = transformer.transform(day_02)
series_03_scaled = transformer.transform(day_03)
val_scaled = transformer.transform(val)
scaler = MinMaxScaler(feature_range=(0, 1))
"""
transformer = Scaler(scaler)
series_01_scaled = transformer.fit_transform(day_01)
series_02_scaled = transformer.fit_transform(day_02)
series_03_scaled = transformer.fit_transform(day_03)
val_scaled = transformer.fit_transform(val)
"""
"""
series_01_scaled.plot()
series_02_scaled.plot()
series_03_scaled.plot()
"""
train_01, val_01 = val_scaled[:-80], val_scaled[-80:]
test, validate = val_scaled[:-80], val_scaled[-80:]
print(val_scaled)
model = BlockRNNModel(
    input_chunk_length=100, output_chunk_length=10, model="RNN"
)
model.fit([test])
#model.fit(biere)
pred = model.predict(series = test, n=80)
val_scaled.plot(label="actual")
#test.plot(new_plot=True)
pred.plot(label="forecast")
print(pred.head)
plt.legend()
#print("MAPE = {:.2f}%".format(mape(series_01_scaled, pred)))
