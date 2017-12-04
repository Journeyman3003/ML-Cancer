import pandas as pd

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from sklearn.preprocessing import Imputer

from statsmodels.imputation.mice import MICEData, MICE
import statsmodels.api as sm

# regression metrics
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
    print(df)
    print(df.info())
    #print(df.describe())


### DATAFRAME OPERATIONS

def writeDFtoCSV(dataFrame, filename):
    dataFrame.to_csv(filename, sep=',', encoding='utf-8')


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


### RANDOM FOREST IMPUTATION FOR MISSING DATA

def randomForestImputation(dataframe, na_threshold=2):

    dataframe = dataframe.loc[:, dataframe.isnull().sum() < dataframe.shape[0] / na_threshold]
    # drop 'NAME' column
    temp = dataframe.drop('NAME', axis=1)
    for column in list(temp):
        featureFrame = temp.drop(column, axis=1)
        targetCol = temp[column]

        nullBoolean = targetCol.isnull().values
        y_test_indexes = targetCol.index[nullBoolean]

        # quick and dirty mean impute the feature frame...
        featureFrame = meanImputation(featureFrame)


        X_train = np.array(featureFrame[list(map(lambda x: not x, nullBoolean))])
        y_train = np.array(targetCol[list(map(lambda x: not x, nullBoolean))])
        # determine most influential 5 features
        X_train = selectBestFeatures(X_train, y_train, 5)

        X_test = selectBestFeatures(X_train, y_train, 5) # np.array(featureFrame[nullBoolean])

        print('#########')
        print(y_test_indexes)
        forest = RandomForestRegr(X_train, y_train)
        pred = forest.predict(X_test)
        for i, missingindex in enumerate(y_test_indexes):
            print('predicted value for column', column, 'and row', missingindex, 'is', pred[i])
            dataframe.at[missingindex, column] = pred[i]

    return dataframe


### PREDICTIVE MEAN MATCHING FOR MISSING DATA/FEATURES

def predictiveMeanMatching(dataframe, na_threshold):

    dataframe = dataframe.loc[:, dataframe.isnull().sum() < dataframe.shape[0]/na_threshold]
    dataframe.columns = map(lambda x: 'c' + str(x), range(dataframe.shape[1]))
    dataframe.rename(columns={'c0': 'NAME'})
    # drop 'NAME' column
    temp = dataframe.drop('NAME', axis=1)
    for column in list(temp):
        featureFrame = temp.drop(column, axis=1)
        targetCol = temp[column]

        #noNaNCols = dataframe.dropna(axis=1, how='any')

        # NaNCols = dataframe.loc[:, df.isnull().any()]

        # second, determine the 4 most influential regressors among them

        # third, MICE

        imp = MICEData(dataframe)
        #print(imp)
        fml = 'c1 ~ c2'
        mice = MICE(fml, sm.OLS, imp)
        results = mice.fit(5, 5)
        print(results.summary())
        # results.


def meanImputation(dataframe):
    """
    plain mean imputation
    :param dataframe:
    :return:
    """
    imp = Imputer(strategy='mean', axis=1)
    temp = pd.DataFrame(imp.fit_transform(dataframe))
    # imp = imp.fit(dataframe)
    # temp = imp.transform(dataframe)
    temp.columns = dataframe.columns
    temp.index = dataframe.index
    #dataframe.fillna(dataframe.mean())

    return temp

def findBestPredictorCols(dataframe, targetCol, nPredictors):

    X = dataframe.drop(targetCol, axis=1)
    y = dataframe[targetCol]
    selectKBest = selectBestFeatures(X,y, nPredictors)
    mask = selectKBest.get_support()
    print(mask)

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

        y_testList.extend(y_test)
        y_predictList.extend(y_predict)

    return y_testList, y_predictList


def createFeatureAndTargetArrays(dataframe, targetCol):

    X = np.array(dataframe.drop(targetCol, axis=1))
    y = np.array(dataframe[targetCol])

    return X, y


def predictDrugPerformance(df, nFeatures=30):
    """

    :param df: imputed and already filtered for a particular cell line, holds all features to use
    :param nFeatures:
    :return:
    """

    X, y = createFeatureAndTargetArrays(df, 'GROWTH')

    # select only best features to use
    X_best = selectBestFeatures(X, y, nFeatures)

    # five-fold cross-validation
    y_test, y_predict = kFoldCrossValidation(X_best, y, 5)

    return y_test, y_predict,


# def run(multiConcentration=False, save=False)



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
    doseResponseFitted = loadDoseResponseFitted()
    summarizeDF(doseResponseFitted)

    drugDescriptors = loadDrugDescriptors()
    summarizeDF(drugDescriptors)

    # dfdescriptors = randomForestImputation(drugDescriptors,2)
    # writeDFtoCSV(dfdescriptors, 'randomForestImputedDescriptors.csv')
    #
    # # predictiveMeanMatching(df2, na_threshold=2)
    ECPF1024 = loadECPF1024()
    summarizeDF(ECPF1024)
    #    #
    # # merge = mergeDataFrames(doseResponseFitted, 'NSC',(drugDescriptors, 'NAME'),(ECPF1024, 'DRUG'))
    # # summarizeDF(merge)
    #
    PFP1024 = loadPFP1024()
    summarizeDF(PFP1024)
    #
    # doseResponseMulti = loadDoseResponseMultipleDoses()
    # summarizeDF(doseResponseMulti)

    ### IMPUTED DESCRIPTORS

    # mean
    meanImputedDescriptors = meanImputation(drugDescriptors)

    # random forest
    # forestImputedDescriptors = randomForestImputation(drugDescriptors, 2)
    forestImputedDescriptors = loadFileToDf('randomForestImputedDescriptors.csv', '', separator=',')

    """
    ### DEFAULT MODEL:
    """

    # just descriptors as features
    cell683665 = filterDataFrame(doseResponseFitted, 'CELLNAME', 683665)
    merge = mergeDataFrames(cell683665, 'NSC', (meanImputedDescriptors, 'NAME'))
    print(merge.shape)
    merge.drop(['NSC', 'NAME', 'CELLNAME'], axis=1, inplace=True)
    for i in [1,10,100,1000]:
        y_test, y_predict = predictDrugPerformance(merge, i)
        r2 = r2_score(y_test,y_predict)
        mse = mean_squared_error(y_test,y_predict)
        print(i, 'FEATURES:\nR2:', r2, 'MSE:',mse)

    # just Fingerprint information - ECPF
    # cell683665 = filterDataFrame(doseResponseFitted, 'CELLNAME', 683665)
    # merge = mergeDataFrames(cell683665, 'NSC', (ECPF1024, 'DRUG'))
    # print(merge.shape)
    # merge.drop(['NSC', 'DRUG', 'CELLNAME'], axis=1, inplace=True)
    # for i in [1, 10, 100, 1000]:
    #     y_test, y_predict = predictDrugPerformance(merge, i)
    #     r2 = r2_score(y_test, y_predict)
    #     mse = mean_squared_error(y_test, y_predict)
    #     print(i, 'FEATURES:\nR2:', r2, 'MSE:', mse)

    # just Fingerprint information - PFP
    # cell683665 = filterDataFrame(doseResponseFitted, 'CELLNAME', 683665)
    # merge = mergeDataFrames(cell683665, 'NSC', (PFP1024, 'DRUG'))
    # print(merge.shape)
    # merge.drop(['NSC', 'DRUG', 'CELLNAME'], axis=1, inplace=True)
    # for i in [1, 10, 100, 1000]:
    #     y_test, y_predict = predictDrugPerformance(merge, i)
    #     r2 = r2_score(y_test, y_predict)
    #     mse = mean_squared_error(y_test, y_predict)
    #     print(i, 'FEATURES:\nR2:', r2, 'MSE:', mse)

    # ALL OF IT
    # cell683665 = filterDataFrame(doseResponseFitted, 'CELLNAME', 683665)
    # merge = mergeDataFrames(cell683665, 'NSC', (meanImputedDescriptors, 'NAME'), (ECPF1024, 'DRUG'),(PFP1024, 'DRUG'))
    # print(merge.shape)
    # merge.drop(['NSC', 'DRUG_x', 'DRUG_y', 'NAME', 'CELLNAME'], axis=1, inplace=True)
    # for i in [1, 10, 100, 1000]:
    #     y_test, y_predict = predictDrugPerformance(merge, i)
    #     r2 = r2_score(y_test, y_predict)
    #     mse = mean_squared_error(y_test, y_predict)
    #     print(i, 'FEATURES:\nR2:', r2, 'MSE:', mse)

    """
    ### FOREST IMPUTED:
    """

    # just descriptors as features
    cell683665 = filterDataFrame(doseResponseFitted, 'CELLNAME', 683665)
    merge = mergeDataFrames(cell683665, 'NSC', (forestImputedDescriptors, 'NAME'))
    print(merge.shape)
    merge.drop(['NSC', 'NAME', 'CELLNAME'], axis=1, inplace=True)
    for i in [1,10,100,1000]:
        y_test, y_predict = predictDrugPerformance(merge, i)
        r2 = r2_score(y_test,y_predict)
        mse = mean_squared_error(y_test,y_predict)
        print(i, 'FEATURES:\nR2:', r2, 'MSE:',mse)








