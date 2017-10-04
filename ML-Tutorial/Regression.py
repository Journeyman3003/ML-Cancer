import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# creates an actual dataframe object
df = quandl.get("WIKI/GOOGL")

# print(df.head())

# reduce df to relevant columns
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

# add new column 'HL_PCT' to reflect high-low volatility
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0

# add new column 'PCT_change' to reflect daily percent change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
# print(df.head())


# Actual machine learning

# label column
forecast_col = 'Adj. Close'

# remove N/A values
# replacing it with such a drastic value will make it an outlier-value
# to be ignored by ML algorithms
df.fillna(value=-99999, inplace=True)

# We're saying we want to forecast out 1%
# of the entire length of the dataset.
# Thus, if our data is 100 days of stock prices,
# we want to be able to predict the price
# 1 day out into the future.

# forecast_out = only the AMOUNT of data/days to be predicted
# currently 34
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
# print(df)

# drop NaN
# inplace flag defines dataframe dest = dataframe src
# drops rows with NaN
df.dropna(inplace=True)
print(df)

# split dataframe /now converted numpy array into features X
# and outcomes y

# features (now numpy array)
X = np.array(df.drop(['label'], 1))
# outcomes (now numpy array)
y = np.array(df['label'])
