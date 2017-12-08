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

from scipy import stats

import numpy as np
import os
import logging

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

    df.rename(columns={'COSMIC identifier': 'CELLNAME'}, inplace=True)

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

def selectBestFeatures(X, y, nFeatures, transform=True):

    print('Choosing best', nFeatures, 'features out of the given ones...')
    if transform:
        return SelectKBest(f_regression, k=nFeatures).fit_transform(X, y)
    else:
        return SelectKBest(f_regression, k=nFeatures).fit(X, y)


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
        featureFrame = meanImputation(featureFrame, axis=0)

        print('quick and dirty imputed feature frame shape:', featureFrame.shape)


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

    # rename columns due to '#' symbols (python comment) in column names...
    dataframe.columns = map(lambda x: 'c' + str(x), range(dataframe.shape[1]))
    dataframe.rename(columns={'c0': 'NAME'}, inplace=True)
    # drop 'NAME' column
    temp = dataframe.drop('NAME', axis=1)
    for column in list(temp):
        featureFrame = temp.drop(column, axis=1)
        targetCol = temp[column]

        # quick and dirty mean impute the feature frame...
        featureFrame = meanImputation(featureFrame, axis=0)

        # determine NaN indexes in target col
        nullBoolean = targetCol.isnull().values
        NaN_indexes = targetCol.index[nullBoolean]

        featureFrame_noNaN = featureFrame.drop(NaN_indexes, axis=0)
        targetCol_noNaN = targetCol.drop(NaN_indexes, axis=0)
        # determine the 4 most influential regressors among them
        bestSelectedFeatures = selectBestFeatures(featureFrame_noNaN, targetCol_noNaN, 5, transform=False)

        mask = bestSelectedFeatures.get_support()  # list of booleans
        new_features = []  # The list of k best features

        for bool, feature in zip(mask, list(featureFrame)):
            if bool:
                new_features.append(feature)

        print(new_features)

        # prepare lm formula string
        betas = '+'.join(new_features)
        formula = column + ' ~ ' + betas
        print(formula)

        #rejoin dataframe
        miceframe = pd.merge(targetCol.reset_index(), featureFrame[new_features], left_index=True, right_index=True)
        miceframe.drop('index', axis=1, inplace=True)
        summarizeDF(miceframe)

        # third, MICE
        try:
            imp = MICEData(miceframe)
            mice = MICE(formula, sm.OLS, imp)
            results = mice.fit(5, 5)
            print(results.summary())
        except ValueError:
            logging.exception("message")
        # TODO finish, but obsolete since package has bugs.... :-(


def meanImputation(dataframe, axis=1):
    """
    plain mean imputation
    :param dataframe:
    :return:
    """
    imp = Imputer(strategy='mean', axis=axis)
    temp = pd.DataFrame(imp.fit_transform(dataframe))
    # imp = imp.fit(dataframe)
    # temp = imp.transform(dataframe)
    temp.columns = dataframe.columns
    temp.index = dataframe.index
    #dataframe.fillna(dataframe.mean())

    return temp


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


def predictDrugPerformance(df, nFeatures=100):
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

    return y_test, y_predict


def createCellLineResponseDict(drugResponseData, *featureDataframesAndMergeCols, nFeatures=100, maxCL=0,
                               cellLinesList=None):

    cellLineResponseDict = {}

    # first, determine unique cellLine COSMIC IDs
    if cellLinesList is not None:
        cosmic = cellLinesList
    else:
        cosmic = drugResponseData['CELLNAME'].unique()
        # if evalLines >0, take only first x cellLines into account
        if maxCL > 0:
            cosmic = cosmic[:maxCL]

    for i, cellLine in enumerate(cosmic):
        print(' #####################################\n',
              '## RUN', i, '-> CELL LINE ', cellLine, '##\n',
              '#####################################')

        cellLineResponseDict[cellLine] = {}

        # filter response data for given cell line
        filterResponse = filterDataFrame(drugResponseData, 'CELLNAME', cellLine)
        #drop CELLNAME
        filterResponse.drop('CELLNAME', axis=1, inplace=True)


        # merge target and features
        merge = mergeDataFrames(filterResponse, 'NSC', *featureDataframesAndMergeCols)
        # drop all merge columns (all possible ones that could be still in the merge, ignore errors)
        merge.drop(['NSC', 'DRUG', 'DRUG_x', 'DRUG_y', 'NAME'], axis=1, errors='ignore', inplace=True)

        # predict all (5 fold cross validation)
        y_test, y_predict = predictDrugPerformance(merge, nFeatures)

        # compute metrics
        r2 = r2_score(y_test, y_predict)
        mse = mean_squared_error(y_test, y_predict)

        # record in dict
        cellLineResponseDict[cellLine]['r2'] = r2
        cellLineResponseDict[cellLine]['mse'] = mse

    return cellLineResponseDict


def evaluateParameterPerformance(resultDict, xAxisLabel, yAxisLabel, barplot=False, xTicks=None, xLabels='',
                                 plotTitle='', saveName=''):
    fig = plt.figure()

    plt.xlabel(xAxisLabel)
    plt.ylabel(yAxisLabel)

    if barplot:
        #start at 1 thus +1 as limit
        ind = np.arange(1, len(resultDict.keys())+1)

        plt.xticks(ind, resultDict.keys(), rotation='horizontal')

        i = 0
        for key, value in resultDict.items():
            plt.bar(ind[i], value, color="rbgkm"[i], label=key)
            i += 1
    else:
        #counter for rgb
        i = 0
        for key, value in resultDict.items():
            plt.plot(value['x'], value['y'], '--^', color="rbgkm"[i], linewidth=0.5, label=key)
            i += 1
        if xTicks is not None:
            plt.xticks(xTicks, xLabels, rotation='vertical')
    plt.title(plotTitle)

    plt.legend(loc='upper right')

    if saveName != "":
        plt.savefig(saveName+'.png', format='png', bbox_inches='tight', dpi=100)
    #plt.show()
    plt.close()


def evaluateByCellLinePerformance(*resultFileAndSeriesNames, metric='r2', saveName=''):
    #load from file

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)

    plt.title('Performance of by-cell line regression models')
    plt.xlabel('Cell Lines (COSMIC ID)')

    if metric=='r2':
        plt.ylabel('R2 Score')
    else:
        plt.ylabel('Mean Squared Error')

    y=[]
    y2=[]

    i=0
    x=[]
    for resultfile, seriesName in resultFileAndSeriesNames:
        cellLineDict = loadPythonObjectFromFile(resultfile)

        results = [(key, value[metric]) for key, value in cellLineDict.items()]
        print('SERIES', i)

        print(metric, results)
        # process first (main) series differently: defines the order
        if i==0:
            if metric=='r2':
                # only reverse order in case of r2
                sortedResults = sorted(results, key=lambda tuple: tuple[1], reverse=True)
            else:
                sortedResults = sorted(results, key=lambda tuple: tuple[1], reverse=False)

            print(sortedResults)

            x, y = zip(*sortedResults)
            plt.plot(range(len(y)), y, '--^', color="rbgkm"[i], linewidth=0.5, label=seriesName)
            plt.xticks(range(len(x)), x, rotation='vertical')
        else:
            temp = []
            print('reults',results)
            for cosmicID in x:
                temp.append([pair for pair in results if pair[0] == cosmicID][0])
            _, y2 = zip(*temp)

            plt.plot(range(len(y2)), y2, '--^', color="rbgkm"[i], linewidth=0.5, label=seriesName)


        i+=1

    # compute spearman rank correlation
    comparisonrank = stats.rankdata(y, method='average')

    predictedrank = stats.rankdata(y2, method='average')

    spearmanRCol, pvalue = stats.spearmanr(comparisonrank, predictedrank)

    plt.text(0.02, 0.2,
             'Spearman Rank Correlation: ' + str(round(spearmanRCol, 4)) + '\np-value: ' + format(pvalue, '.3g'),
             horizontalalignment='left',
             verticalalignment='center',
             transform=ax.transAxes,
             bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

    plt.legend(loc='upper right')
    ax.axhline(y=0, color='black', ls='--', lw=0.7)

    if saveName != "":
        plt.savefig(saveName+'.png', format='png', bbox_inches='tight', dpi=100)
        plt.close()
    else:
        plt.show()



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
    # yes, deactivating warnings might sometimes be a bad idea, but this time it should be fine...
    pd.options.mode.chained_assignment = None

    doseResponseFitted = loadDoseResponseFitted()
    summarizeDF(doseResponseFitted)

    drugDescriptors = loadDrugDescriptors()
    summarizeDF(drugDescriptors)

    ECPF1024 = loadECPF1024()
    summarizeDF(ECPF1024)
    #
    PFP1024 = loadPFP1024()
    summarizeDF(PFP1024)
    #
    doseResponseMulti = loadDoseResponseMultipleDoses()
    summarizeDF(doseResponseMulti)

    ### IMPUTED DESCRIPTORS

    # mean
    meanImputedDescriptors = meanImputation(drugDescriptors)

    # random forest
    # forestImputedDescriptors = randomForestImputation(drugDescriptors, 2)
    forestImputedDescriptors = loadFileToDf('randomForestImputedDescriptors.csv', '', separator=',')


    # PMM
    """
    Please note: I tried to use MICE to perform predictive mean matching. Unlike in R, there is no 1.0 version of a
    package to perform MICE. The statsmodel package that I tried to use seems to have some severe bugs..., i.e. when 
    there is more than one value to impute, it crashes. I don't have the time to fix that bug, thus I could not make
    use of MICE imputation
    """
    # pmmImputedDescriptors = predictiveMeanMatching(drugDescriptors,2)
    """
    ### DEFAULT MODEL:
    ### determine number of features to use
    """
    featuresToTest = [1,10,100,1000]
    r2_scores = {}
    mse_values = {}

    cellsToTest = [683665, 909774, 949160, 749716, 1240210]

    for cell in cellsToTest:
        # just descriptors as features
        r2_scores['only drug descriptors'] = {}
        r2_scores['only drug descriptors']['x'] = featuresToTest
        r2_scores['only drug descriptors']['y'] = []

        mse_values['only drug descriptors'] = {}
        mse_values['only drug descriptors']['x'] = featuresToTest
        mse_values['only drug descriptors']['y'] = []

        cell683665 = filterDataFrame(doseResponseFitted, 'CELLNAME', cell)
        merge = mergeDataFrames(cell683665, 'NSC', (meanImputedDescriptors, 'NAME'))
        print(merge.shape)
        merge.drop(['NSC', 'NAME', 'CELLNAME'], axis=1, inplace=True)
        for i in featuresToTest:
            y_test, y_predict = predictDrugPerformance(merge, i)
            r2 = r2_score(y_test,y_predict)
            mse = mean_squared_error(y_test,y_predict)
            r2_scores['only drug descriptors']['y'].append(r2)
            mse_values['only drug descriptors']['y'].append(mse)
            print(i, 'FEATURES:\nR2:', r2, 'MSE:',mse)

        # just Fingerprint information - ECPF

        r2_scores['only ECPF'] = {}
        r2_scores['only ECPF']['x'] = featuresToTest
        r2_scores['only ECPF']['y'] = []

        mse_values['only ECPF'] = {}
        mse_values['only ECPF']['x'] = featuresToTest
        mse_values['only ECPF']['y'] = []

        cell683665 = filterDataFrame(doseResponseFitted, 'CELLNAME', cell)
        merge = mergeDataFrames(cell683665, 'NSC', (ECPF1024, 'DRUG'))
        print(merge.shape)
        merge.drop(['NSC', 'DRUG', 'CELLNAME'], axis=1, inplace=True)
        for i in featuresToTest:
            y_test, y_predict = predictDrugPerformance(merge, i)
            r2 = r2_score(y_test, y_predict)
            mse = mean_squared_error(y_test, y_predict)

            r2_scores['only ECPF']['y'].append(r2)
            mse_values['only ECPF']['y'].append(mse)
            print(i, 'FEATURES:\nR2:', r2, 'MSE:', mse)


        # just Fingerprint information - PFP

        r2_scores['only PFP'] = {}
        r2_scores['only PFP']['x'] = featuresToTest
        r2_scores['only PFP']['y'] = []

        mse_values['only PFP'] = {}
        mse_values['only PFP']['x'] = featuresToTest
        mse_values['only PFP']['y'] = []

        cell683665 = filterDataFrame(doseResponseFitted, 'CELLNAME', cell)
        merge = mergeDataFrames(cell683665, 'NSC', (PFP1024, 'DRUG'))
        print(merge.shape)
        merge.drop(['NSC', 'DRUG', 'CELLNAME'], axis=1, inplace=True)
        for i in featuresToTest:
            y_test, y_predict = predictDrugPerformance(merge, i)
            r2 = r2_score(y_test, y_predict)
            mse = mean_squared_error(y_test, y_predict)

            r2_scores['only PFP']['y'].append(r2)
            mse_values['only PFP']['y'].append(mse)

            print(i, 'FEATURES:\nR2:', r2, 'MSE:', mse)

        # ALL OF IT

        r2_scores['all combined'] = {}
        r2_scores['all combined']['x'] = featuresToTest
        r2_scores['all combined']['y'] = []

        mse_values['all combined'] = {}
        mse_values['all combined']['x'] = featuresToTest
        mse_values['all combined']['y'] = []

        cell683665 = filterDataFrame(doseResponseFitted, 'CELLNAME', cell)
        merge = mergeDataFrames(cell683665, 'NSC', (meanImputedDescriptors, 'NAME'), (ECPF1024, 'DRUG'),(PFP1024, 'DRUG'))
        print(merge.shape)
        merge.drop(['NSC', 'DRUG_x', 'DRUG_y', 'NAME', 'CELLNAME'], axis=1, inplace=True)
        for i in featuresToTest:
            y_test, y_predict = predictDrugPerformance(merge, i)
            r2 = r2_score(y_test, y_predict)
            mse = mean_squared_error(y_test, y_predict)

            r2_scores['all combined']['y'].append(r2)
            mse_values['all combined']['y'].append(mse)

            print(i, 'FEATURES:\nR2:', r2, 'MSE:', mse)

        writePythonObjectToFile(r2_scores, 'r2_based_feature_number_' + str(cell))
        writePythonObjectToFile(mse_values, 'mse_based_feature_number_' + str(cell))

        r2_scores = loadPythonObjectFromFile('r2_based_feature_number_' + str(cell))
        mse_values = loadPythonObjectFromFile('mse_based_feature_number_' + str(cell))

        evaluateParameterPerformance(r2_scores, xAxisLabel='number of features used', yAxisLabel='R2-Score',
                                     saveName='optimalNumberOfFeaturesR2' + str(cell), plotTitle='Optimal number of features to use')
        evaluateParameterPerformance(mse_values, xAxisLabel='number of features used', yAxisLabel='Mean Squared Error',
                                     saveName='optimalNumberOfFeaturesMSE' + str(cell), plotTitle='Optimal number of features to use')

    """
    ### FOREST IMPUTED:
    """

    r2_scores = {}
    mse_values = {}
    cellsToTest = [683665, 909774, 949160, 749716, 1240210]
    # just descriptors as features

    for cellLine in cellsToTest:
        # just mean imputed descriptors as features
        cell683665 = filterDataFrame(doseResponseFitted, 'CELLNAME', cellLine)
        merge = mergeDataFrames(cell683665, 'NSC', (meanImputedDescriptors, 'NAME'))
        print(merge.shape)
        merge.drop(['NSC', 'NAME', 'CELLNAME'], axis=1, inplace=True)

        y_test, y_predict = predictDrugPerformance(merge, 100)
        r2 = r2_score(y_test, y_predict)
        mse = mean_squared_error(y_test, y_predict)
        r2_scores['mean imputed'] = r2
        mse_values['mean imputed'] = mse
        print('\nR2:', r2, 'MSE:', mse)

        # just rf imputed descriptors as features

        cell683665 = filterDataFrame(doseResponseFitted, 'CELLNAME', cellLine)
        merge = mergeDataFrames(cell683665, 'NSC', (forestImputedDescriptors, 'NAME'))
        print(merge.shape)
        merge.drop(['NSC', 'NAME', 'CELLNAME'], axis=1, inplace=True)

        y_test, y_predict = predictDrugPerformance(merge, 100)
        r2 = r2_score(y_test,y_predict)
        mse = mean_squared_error(y_test,y_predict)
        r2_scores['random forest imputed'] = r2
        mse_values['random forest imputed'] = mse
        print('\nR2:', r2, 'MSE:',mse)

        evaluateParameterPerformance(r2_scores, xAxisLabel='different imputation metrics', yAxisLabel='R2-Score',
                                     barplot=True, plotTitle='Comparison of imputation techniques',
                                     saveName='imputationComparisonR2' + str(cellLine))

        evaluateParameterPerformance(mse_values, xAxisLabel='different imputation metrics', yAxisLabel='Mean Squared Error',
                                     barplot=True, plotTitle='Comparison of imputation techniques',
                                     saveName='imputationComparisonMSE' + str(cellLine))


    """
    now, after params are determined: evaluate performance on several cell Line models
    """

    dictT = loadPythonObjectFromFile('cellLineDict_100cells_PFPonly_multiple')

    lineList = list(dictT.keys())
    lineList = lineList[:80]

    writePythonObjectToFile(lineList,'100CellLines')

    cellLinesList = loadPythonObjectFromFile('100CellLines')

    # using the fitted responses
    cellLineDict = createCellLineResponseDict(doseResponseFitted, (forestImputedDescriptors, 'NAME'),
                                              (ECPF1024, 'DRUG'),(PFP1024, 'DRUG'), nFeatures=100, maxCL=100,
                                              cellLinesList=cellLinesList)

    print(cellLineDict)

    writePythonObjectToFile(cellLineDict, 'cellLineDict_100cells_PFPonly_fitted')

    # using multiple measured doses
    cellLineDict = createCellLineResponseDict(doseResponseMulti, (forestImputedDescriptors, 'NAME'),
                                              (ECPF1024, 'DRUG'),(PFP1024, 'DRUG'), nFeatures=100, maxCL=100,
                                              cellLinesList=cellLinesList)

    print(cellLineDict)

    writePythonObjectToFile(cellLineDict, 'cellLineDict_100cells_PFPonly_multiple')

    evaluateByCellLinePerformance(('cellLineDict_100cells_PFPonly_fitted', 'fitted drug response'),
                                  ('cellLineDict_100cells_PFPonly_multiple', 'multiple doses drug response'),
                                  metric='r2', saveName='PerformanceFitted80CellsR2')

    evaluateByCellLinePerformance(('cellLineDict_100cells_PFPonly_fitted', 'fitted drug response'),
                                  ('cellLineDict_100cells_PFPonly_multiple', 'multiple doses drug response'),
                                  metric='mse', saveName='PerformanceFitted80CellsMSE')









