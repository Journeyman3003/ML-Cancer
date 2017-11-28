import pandas as pd
import os
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

import numpy as np

# regression metrics
from sklearn.metrics import r2_score, mean_squared_error

# directory where to find data
dataPath = 'data'


def summarizeDF(df):
    print(df)
    print('df.shape:', df.shape)
    print(df.info())
    print(df.describe())


def loadFileToDf(fileName, separator=',', skip=0, nrows=None, header=0, encoding=None):

    filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath, fileName)
    dataFrame = pd.read_csv(filePath, sep=separator, skiprows=skip, nrows=nrows, header=header, encoding=encoding)

    return dataFrame


def loadMelanomaFrame(summarize=False):
    melanomaFrame = loadFileToDf('melanoma-v08.txt', separator='\t')
    # summarizeDF(melanomaFrame)
    melanomaFrame = melanomaFrame[[' FIPS', 'Age-Adjusted Incidence Rate']]

    melanomaFrame['Age-Adjusted Incidence Rate'].replace('* ', np.NaN, inplace=True, axis=1, regex=False)

    if summarize:
        summarizeDF(melanomaFrame)

    return melanomaFrame


def loadIncomeFrame(summarize=False):
    incomeFrame = loadFileToDf('median_income.csv', skip=6, nrows=3142)
    # summarizeDF(incomeFrame)
    incomeFrame = incomeFrame[[' FIPS', 'Value (Dollars)']]

    if summarize:
        summarizeDF(incomeFrame)

    return incomeFrame


def loadSunlinghtFrame(all=True, summarize=False):
    if all:
        sunlightFrame = loadFileToDf('NLDAS_Daily_Sunlight_1979-2011.txt', separator='\t', nrows=3111,
                                     encoding='iso-8859-1')
    else:
        sunlightFrame = loadFileToDf('NLDAS_Daily_Sunlight_2000-2011.txt', separator='\t', nrows=3111,
                                     encoding='iso-8859-1')
    sunlightFrame.rename(columns={'Avg Daily Sunlight (KJ/m²)': 'Avg Daily Sunlight'}, inplace=True)
    sunlightFrame = sunlightFrame[['County Code', 'Avg Daily Sunlight']]##, 'Min Daily Sunlight', 'Max Daily Sunlight']]

    if summarize:
        summarizeDF(sunlightFrame)

    return sunlightFrame


# based on file
# county_adjacency.txt
def loadCountyAdjacencyFrame(summarize=False):
    """
    additional features: country melanoma rates based on adjacency of counties
    :return: dataframe holding
    """
    df = loadFileToDf('county_adjacency.txt', separator='\t', header=None, encoding='iso-8859-1')
    df = df[[1, 3]]
    df.columns = ['base_FIPS', 'adj_FIPS']
    # df.columns = ['base_county', 'base_FIPS', 'adj_county', 'adj_FIPS']
    # df = df[['base_FIPS', 'adj_FIPS']]
    df = df.fillna(method='ffill')
    df['base_FIPS'] = df['base_FIPS'].astype(int)

    if summarize:
        summarizeDF(df)

    return df


def mergeDataFrames(mergeColumn, mainFrame, *dataframes):

    for frame in dataframes:
        mainFrame = mainFrame.merge(frame[0], how='inner', left_on=mergeColumn, right_on=frame[1])

    return mainFrame


def createAdjacencyMelanomaFrame(melanomaFrame, adjacencyFrame, imputeStrategy=None):
    # base counties
    #merge = mergeDataFrames(' FIPS', melanomaFrame, (adjacencyFrame, 'base_FIPS'))
    #merge.drop(' FIPS', inplace=True, axis=1)
    #merge.rename(columns={'Age-Adjusted Incidence Rate': 'base_incident_rate'}, inplace=True)

    # adjacent
    merge = mergeDataFrames(' FIPS', melanomaFrame, (adjacencyFrame, 'adj_FIPS'))
    merge.drop([' FIPS', 'adj_FIPS'], inplace=True, axis=1)
    merge.rename(columns={'Age-Adjusted Incidence Rate': 'adj_incident_rate'}, inplace=True)
    merge.rename(columns={'Age-Adjusted Incidence Rate': 'adj_incident_rate'}, inplace=True)

    #
    if imputeStrategy is not None:
        imp = Imputer(missing_values='NaN', strategy=imputeStrategy, axis=0)
        temp = pd.DataFrame(imp.fit_transform(merge))
        temp.columns = merge.columns
        temp.index = merge.index
        merge = temp
    else:
        merge.dropna(axis=0, inplace=True)

    summarizeDF(merge)
    # compute metrics: average, max and min incidence rate

    #average = merge.groupby('base_FIPS', as_index=False, axis=1)['adj_incident_rate'].mean().add_prefix('mean_')
    #maxx = merge.groupby('base_FIPS', as_index=False, axis=1)['adj_incident_rate'].max().add_prefix('max_')
    #minn = merge.groupby('base_FIPS', as_index=False, axis=1)['adj_incident_rate'].max().add_prefix('min_')

    merge['base_FIPS'] = merge['base_FIPS'].astype(int)

    df = merge.groupby('base_FIPS', as_index=True)['adj_incident_rate'].agg({'Mean': 'mean', 'Max':'max', 'Min':'min'}).reset_index()
    print('df:', df)
    #df = mergeDataFrames('base_FIPS', average, (maxx, 'base_FIPS'), (minn, 'base_FIPS'))

    return df


def createFeatureAndLabelArrays(melanomaFrame, incomeFrame, sunlightFrame, adjacencyFrame=None, imputeStrategy=None,
                                summarize=False):

    merge = mergeDataFrames(' FIPS', melanomaFrame, (incomeFrame, ' FIPS'), (sunlightFrame, 'County Code'))
    if adjacencyFrame is not None:
        merge = mergeDataFrames(' FIPS', melanomaFrame, (incomeFrame, ' FIPS'), (sunlightFrame, 'County Code'),
                                (adjacencyFrame, 'base_FIPS'))
        merge.drop(['base_FIPS'], inplace=True, axis=1)
    merge.drop(['County Code',' FIPS'], inplace=True, axis=1)

    if imputeStrategy is not None:
        imp = Imputer(missing_values='NaN', strategy=imputeStrategy, axis=0)
        temp = pd.DataFrame(imp.fit_transform(merge))
        temp.columns = merge.columns
        temp.index = merge.index
        merge = temp
    else:
        merge.dropna(axis=0, inplace=True)

    if summarize:
        summarizeDF(merge)

    X = np.array(merge.drop('Age-Adjusted Incidence Rate', axis=1))
    y = np.array(merge['Age-Adjusted Incidence Rate'], dtype=np.float32)

    return X, y


def RandomForestRegr(X, y):
    # 500 trees, all cores
    forestRegr = RandomForestRegressor(n_estimators=500, criterion="mse", n_jobs=-1)
    forestRegr.fit(X, y)

    return forestRegr


def LinearRegr(X, y, normalize=False):

    linReg = LinearRegression(normalize=normalize,n_jobs=-1)
    linReg.fit(X,y)

    return linReg


def kFoldCrossValidation(X, y, k, algorithm):
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

        forest = algorithm(X_train, y_train)
        print('done building regressor')
        y_predict = forest.predict(X_test)

        # print('labels:', y_test.shape[0], '\n', y_test)
        # print('predicted labels:', y_test.shape[0], '\n', y_predict)

        y_testList.extend(y_test)
        y_predictList.extend(y_predict)

    return y_testList, y_predictList


def evaluateModel(y_test, y_predict):
    print('### R2 Score: ###')
    print(r2_score(y_test,y_predict))

    print('### MSE Score: ###')
    print(mean_squared_error(y_test, y_predict))


if __name__ == "__main__":
    # base dataframe
    melanomaData = loadMelanomaFrame()

    #print(melanomaData['Age-Adjusted Incidence Rate'].unique())

    # income data
    incomeData = loadIncomeFrame()

    # sunlight data
    sunlightData = loadSunlinghtFrame(all=True)

    # Task 2: county adjacency information
    adjacencyData = loadCountyAdjacencyFrame(summarize=True)

    adjacencyMelanomaData = createAdjacencyMelanomaFrame(melanomaFrame=melanomaData, adjacencyFrame=adjacencyData,
                                                         imputeStrategy='mean')
    print(adjacencyMelanomaData.shape)



    # # Task 1
    # X, y = createFeatureAndLabelArrays(melanomaFrame=melanomaData, incomeFrame=incomeData, sunlightFrame=sunlightData,
    #                                    imputeStrategy=None, summarize=True)
    #
    # # 5-fold cross validation
    #
    # y_test, y_predict = kFoldCrossValidation(X, y, 5, LinearRegr)
    #
    # y_predict = list(map(lambda x: round(x, 2), y_predict))
    #
    # print(y_test)
    # print(y_predict)
    #
    # evaluateModel(y_test=y_test, y_predict=y_predict)
    #
    #
    # Task 2

    X, y = createFeatureAndLabelArrays(melanomaFrame=melanomaData, incomeFrame=incomeData, sunlightFrame=sunlightData,
                                       adjacencyFrame=adjacencyMelanomaData, imputeStrategy='mean', summarize=True)

    # 5-fold cross validation

    y_test, y_predict = kFoldCrossValidation(X, y, 5, LinearRegr)

    #y_predict = list(map(lambda x: round(x, 2), y_predict))

    print(y_test)
    print(y_predict)

    evaluateModel(y_test=y_test, y_predict=y_predict)
    #
    # melanomaFrame = loadFileToDf('melanoma-v08.txt', separator='\t')
    # summarizeDF(melanomaFrame)
    #
    # melanomaFrame = melanomaFrame[['County', ' FIPS', 'Age-Adjusted Incidence Rate']]
    # summarizeDF(melanomaFrame)

    #incomeFrame = loadFileToDf('median_income.csv', skip=6, nrows=3142)
    #summarizeDF(incomeFrame)

    # print(incomeFrame['County'].unique())
    #
    # 1979-2011
    # sunlightFrame1 = loadFileToDf('NLDAS_Daily_Sunlight_1979-2011.txt', separator='\t', nrows=3111, encoding='iso-8859-1')
    # sunlightFrame1.rename(columns={'Avg Daily Sunlight (KJ/m²)': 'Avg Daily Sunlight'}, inplace=True)
    # sunlightFrame1 = sunlightFrame1[['County', 'County Code', 'Avg Daily Sunlight', 'Min Daily Sunlight', 'Max Daily Sunlight']]
    # # 2000-2011
    # sunlightFrame2 = loadFileToDf('NLDAS_Daily_Sunlight_2000-2011.txt', separator='\t', nrows=3111, encoding='iso-8859-1')
    #
    # summarizeDF(sunlightFrame1)
    #
    #
    # # the space though...
    # merge = mergeDataFrames(' FIPS', melanomaFrame, (incomeFrame,' FIPS'), (sunlightFrame1,'County Code'))
    #
    # summarizeDF(merge)

