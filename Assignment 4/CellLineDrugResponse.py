import pandas as pd

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from statsmodels.imputation.mice import MICE

from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

import numpy as np
import os

# to deserialize/serialize the results to/from disk
import pickle as pkl

dataRoot = 'MLiC-Lab4'
project = 'GDSC'
drugData = 'GDSC-drug-descriptors-and-fingerprints'


def summarizeDF(df):
    #print(df)
    print(df.info())
    #print(df.describe())


### DATAFRAME OPERATIONS

def loadFileToDf(fileName, directories, separator=',', skip=None, nrows=None, header=0, encoding=None, na=None,
                 coltypes=None):

    filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), *directories, fileName)
    dataFrame = pd.read_csv(filePath, sep=separator, skiprows=skip, nrows=nrows, header=header, encoding=encoding,
                            na_values=na, dtype=coltypes)

    return dataFrame


def filterDataFrame(df, filterColumn, filterValue):
    return df.loc[df[filterColumn] == filterValue]


def loadDoseResponseFitted():

    df = loadFileToDf(fileName='GDSC_dose_response_fitted_cosmic_mimick.csv', directories=[dataRoot, project])

    df['CELLNAME'] = df['CELLNAME'].apply(lambda x: str(x)[7:])
    df['CELLNAME'] = df['CELLNAME'].astype(int)
    df.drop('LOG_CONCENTRATION', axis=1, inplace=True)

    return df


def loadDoseResponseMultipleDoses():

    df = loadFileToDf(fileName='GDSC_dose_response.csv', directories=[dataRoot, project])

    df.drop(['CONCUNIT', 'EXPID'], axis=1, inplace=True)

    # load cell lines information as dict
    # Cell_Lines_Details.csv from Lab 2
    dictFrame = loadFileToDf(fileName='Cell_Lines_Details.csv', directories=[dataRoot, project], separator=',',
                             header=0, nrows=1001)

    dictFrame = dictFrame[['Sample Name', 'COSMIC identifier']]

    # print('df shape before merge:', df.shape)

    # merge looses ~220,000 rows... still 1.7 million are available
    df = mergeDataFrames(df, 'CELLNAME', (dictFrame, 'Sample Name'))

    # print('df shape after merge:', df.shape)

    df.drop(['CELLNAME','Sample Name'], axis=1, inplace=True)

    return df



def loadDrugDescriptors():

    df = loadFileToDf(fileName='descriptors.txt', separator='\t', directories=[dataRoot, project, drugData], na='na')
    # drop index column
    df.drop('No.', axis=1, inplace=True)
    return df


# used manually created csv files due to error on parsing given tsv files
def loadECPF1024():

    df = loadFileToDf(fileName='ECFP_1024_m0-2_b2_c_0.csv', separator=',', directories=[dataRoot, project, drugData],
                      encoding='iso-8859-1', skip=[0, 1], header=None)

    # rename for convenience
    df.rename(columns={0: 'DRUG'}, inplace=True)

    return df


# used manually created csv files due to error on parsing given tsv files
def loadPFP1024():
    df = loadFileToDf(fileName='PFP_1024_m0-6_b2_c_1.csv', separator=',', directories=[dataRoot, project, drugData],
                      encoding='iso-8859-1', skip=[0, 1], header=None)

    # rename for convenience
    df.rename(columns={0: 'DRUG'}, inplace=True)

    return df


def mergeDataFrames(mainFrame, mergeColumn, *dataframes):
    """
    creates INNER JOIN of given dataframes
    :param mergeColumn: column to merge on given main data frame
    :param mainFrame: main data frame
    :param dataframes: tuples/lists consisting of dataframe and merge column information
    :return: the merged frame
    """

    for frame in dataframes:
        mainFrame = mainFrame.merge(frame[0], how='inner', left_on=mergeColumn, right_on=frame[1])

    return mainFrame


### FEATURE PREPARATIONS

def selectBestFeatures(X, y, nFeatures):

    print('Choosing best', nFeatures, 'features out of the given ones...')
    return SelectKBest(f_regression, k=nFeatures).fit_transform(X, y)

### PREDICTIVE MEAN MATCHING FOR MISSING DATA/FEATURES

def predictiveMeanMatching(dataframe):
    # first, find no-nan columns

    # second, determine the 4 most influential regressors among them

    # third, MICE



### PER CELL LINE MODEL

def RandomForestRegr(X, y):
    # 500 trees, all cores
    forestRegr = RandomForestRegressor(n_estimators=500, criterion="mse", n_jobs=-1)
    forestRegr.fit(X, y)

    return forestRegr


def kFoldCrossValidation(X, y, k):
    """
    cross validation with k split points
    :param X: feature array
    :param y: label array
    :param k: number of split points
    :return:
    """
    kFold = KFold(n_splits=k)
    i = 1
    # create two lists that hold the labels and the predictions
    y_testList = []
    y_predictList = []
    for train_index, test_index in kFold.split(X):
        print('CrossValidation Run:', i)
        i += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        forest = RandomForestRegr(X_train, y_train)
        print('done building forest')
        y_predict = forest.predict(X_test)

        # print('labels:', y_test.shape[0], '\n', y_test)
        # print('predicted labels:', y_test.shape[0], '\n', y_predict)

        y_testList.extend(y_test)
        y_predictList.extend(y_predict)

    return y_testList, y_predictList


def predictDrugPerformance(df, nCellLines):
    None




### READ/WRITE PYTHON OBJECT TO FILE

def writePythonObjectToFile(object, filename):
    print('writing object to file...')
    pkl.dump(object, open(filename + ".pkl", "wb"))
    print('done writing to file...')


def loadPythonObjectFromFile(filename):
    print('loading object from file...')
    object = pkl.load( open(filename + ".pkl", "rb"))
    print('done loading from file...')
    return object


"""
GDSC by-cell Line Drug prediction
"""

if __name__ == '__main__':
    df = loadDoseResponseFitted()
    summarizeDF(df)

    df2 = loadDrugDescriptors()
    summarizeDF(df2)

    df3 = loadECPF1024()
    summarizeDF(df3)

    df = filterDataFrame(df, 'CELLNAME', 683665)
    summarizeDF(df)

    df123 = mergeDataFrames(df, 'NSC',(df2, 'NAME'),(df3, 'DRUG'))
    summarizeDF(df123)

    df4 = loadPFP1024()
    summarizeDF(df4)

    df5 = loadDoseResponseMultipleDoses()
    summarizeDF(df5)

