### title:

# get-rich-quick

### description:

Stock price prediction for fun using AutoTS (outdated)

Current project focusses on estimating a single model for multiple time series (in this case multiple days).
The hope is to find a relatively decent model for predicting stock price minute by minute later on in the day
when we know data from earlier in that same day.

A lot of the time series models in AutoTS are local models, they can forecast the future of the dataset that they studied on.
We need global models, they can learn from multiple datasets and apply their model outside of those datasets.

Summary: Develop a model from stock market panel data (cross sectional time-series data)


### how to use the project:

Given .csv file with historical stock data will train a multitude of time series models and select the best performing model for the given data

### include credits:
https://unit8co.github.io/darts/

AutoTS [pypi](https://pypi.org/project/AutoTS/)