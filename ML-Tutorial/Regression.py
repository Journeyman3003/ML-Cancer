import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
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

# preprocess features to be within a range of [-1,1]
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create support vector regression object
classif = svm.SVR(kernel="linear")

# fit the model given the training data
classif.fit(X_train, y_train)

# test the model given the test data
confidence = classif.score(X_test, y_test)

print(confidence)

# repeat all the stuff with Linear regression

# create simple linear regression classifier
classif2 = LinearRegression()

# fit the model given the training data
classif2.fit(X_train, y_train)

# test the model given the test data
confidence2 = classif2.score(X_test, y_test)

print(confidence2)

# threading to speed up runtime

# create simple linear regression classifier
classif3 = LinearRegression(n_jobs=-1)

# fit the model given the training data
classif3.fit(X_train, y_train)

# test the model given the test data
confidence3 = classif3.score(X_test, y_test)

print(confidence3)

# check confidence for possible svm kernels
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)