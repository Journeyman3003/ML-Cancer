import pandas as pd
import os

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import roc_curve, auc

from sklearn.feature_selection import SelectKBest, chi2, f_regression

from sklearn.preprocessing import label_binarize

from scipy import stats
from scipy import interp

# to deserialize/serialize the classifiers to/from disk
import pickle as pkl

from itertools import cycle
from collections import Counter

filepath = 'MLiC-Lab2'

def summarizeDF(df):
    print(df)
    print(df.info())
    # print(df.describe())

def createDrugResponseDataframes(ic50Filename, cellLineFilename, filterAUC=0.0):

    # ic50 file -> dose response data
    ic50FilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),filepath,ic50Filename)
    ic50Data = pd.read_csv(ic50FilePath,sep=',', header=0)

    # drop some unnecessary columns
    ic50Data.drop(['Dataset_version', 'IC50_Results_ID'], axis=1, inplace=True)

    # apply AUC filter if stated
    ic50Data = ic50Data[ic50Data['AUC'] >= filterAUC]


    print('\nIC50 DATA:')
    summarizeDF(ic50Data)

    # cell line file
    cellLinePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),filepath,cellLineFilename)
    cellLineData = pd.read_csv(cellLinePath, sep='\t', header=None)

    cellLineData.rename(columns={ cellLineData.columns[0]: "COSMIC_ID"}, inplace=True)

    print('\nCELL LINE DATA:')
    summarizeDF(cellLineData)

    return ic50Data, cellLineData

def filterAndJoinDataframes(ic50Frame, cellLineFrame, ic50FilterColumn, ic50FilterValue):

    # apply filter
    ic50Frame = ic50Frame[ic50Frame[ic50FilterColumn] == ic50FilterValue]

    # print('\n FILTERED IC50 DATA:')
    # summarizeDF(ic50Frame)

    # join the data frames
    dataFrame = pd.merge(ic50Frame, cellLineFrame, how='inner', on='COSMIC_ID')

    # print('\nJOINED DATA:')
    # summarizeDF(dataFrame)

    return dataFrame

def RandomForestRegr(X, y):
    # 500 trees, all cores
    forestRegr = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    forestRegr.fit(X, y)

    return forestRegr


def RandomForestClassif(X, y):
    # 500 trees, all cores
    forestClassif = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    forestClassif.fit(X, y)

    return forestClassif

def prepareFeatureAndLabelArrays(dataFrame, nFeatures=-1, classification=True):

    #drop unnecessary columns
    dataFrame.drop(['COSMIC_ID', 'DRUG_ID', 'MAX_CONC_MICROMOLAR', 'AUC', 'RMSE'], axis=1, inplace=True)

    # print('\nCOMBINED LABEL AND FEATURE DATA:')
    # summarizeDF(dataFrame)

    X = np.array(dataFrame.drop('LN_IC50', axis=1))
    y = np.array(dataFrame['LN_IC50'])
    if classification:
        y = np.array(convertLabelToClassificationProblem(y))
    if nFeatures > 0:
        X = selectBestFeatures(X, y, nFeatures, classification)

    # print('FeatureArray (X) Dimensions:', X.shape, '\nLabelArray (y) Dimensions:', y.shape)
    # print('X: ', X)
    # print('y: ', y)
    return X, y


def selectBestFeatures(X, y, nFeatures, classification=True):
    print('Choosing best', nFeatures, 'features out of the given ones...')
    if classification:
        X_new = SelectKBest(chi2, k=nFeatures).fit_transform(X, y)
    else:
        X_new = SelectKBest(f_regression(), k=nFeatures).fit_transform(X, y)
    return X_new


def convertLabelToClassificationProblem(y):
    z_scores_y = pd.DataFrame(stats.zscore(y))
    # print('z_scores:\n', z_scores_y)
    labelArray = z_scores_y[0].apply(lambda x: determineSensitivityLabel(x))
    # print('corresponding labels:\n', labelArray)

    return labelArray

def determineSensitivityLabel(z_score):
    if z_score >= 2:
        return "Resistant"
    elif z_score <= -2:
        return "Sensitive"
    else:
        return "Intermediate"


# def garbage(y_test, y_predict):
#     # y = label_binarize(y, classes=['Resistant', 'Intermediate', 'Sensitive'])
#     y_test = label_binarize(y_test, classes=['Resistant', 'Intermediate', 'Sensitive'])
#     y_predict = label_binarize(y_predict, classes=['Resistant', 'Intermediate', 'Sensitive'])
#
#     # n_classes = y.shape[1]
#
#     # n_classes = len(np.unique(y))
#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
#         print('false positive rate:', fpr[i], '\ntrue positive rate:', tpr[i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predict.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#     # Compute macro-average ROC curve and ROC area
#
#     # First aggregate all false positive rates
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
#     # Finally average it and compute AUC
#     mean_tpr /= n_classes
#
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
#     # Plot all ROC curves
#     lw = 2
#     plt.figure()
#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["micro"]),
#              color='deeppink', linestyle=':', linewidth=4)
#
#     plt.plot(fpr["macro"], tpr["macro"],
#              label='macro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["macro"]),
#              color='navy', linestyle=':', linewidth=4)
#
#     colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#     for i, color in zip(range(n_classes), colors):
#         plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                  label='ROC curve of class {0} (area = {1:0.2f})'
#                        ''.format(i, roc_auc[i]))
#
#     plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Some extension of Receiver operating characteristic to multi-class')
#     plt.legend(loc="lower right")
#     plt.show()


def kFoldCrossValidation(X, y, k, classification=True):
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

        if classification:
            forest = RandomForestClassif(X_train, y_train)
        else:
            forest = RandomForestRegr(X_train, y_train)
        print('done building forest')
        y_predict = forest.predict(X_test)

        # print('labels:', y_test.shape[0], '\n', y_test)
        # print('predicted labels:', y_test.shape[0], '\n', y_predict)

        y_testList.extend(y_test)
        y_predictList.extend(y_predict)

    return y_testList, y_predictList


def evaluatePerformance(y_test, y_predict):
    # accuracy:
    accuracy = accuracy_score(y_test, y_predict)
    print('accuracy: ', accuracy)

    # F1 score
    f1 = f1_score(y_test,y_predict,average='macro')
    print('f1: ', f1)

    # AUC Score
    computeMacroAuc(y_test, y_predict)

def computeMacroAuc(y_test, y_predict):
    # Binarize the output
    print('Counter for test data (actual labels): ', Counter(list(y_test)))
    print('Counter for predicted data (predicted labels): ', Counter(list(y_predict)))
    y_test = label_binarize(y_test, classes=['Resistant', 'Intermediate', 'Sensitive'])
    y_predict = label_binarize(y_predict, classes=['Resistant', 'Intermediate', 'Sensitive'])
    n_classes = y_test.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw=2
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def createDrugResponseDict(ic50, cellLine, filterField, nFeatures=-1, classification=True):

    drugResponseDict = {}
    for i, drug in enumerate(ic50[filterField].unique()):
        print(' #####################################\n',
              '##             RUN ',i,'             ##\n',
              '#####################################')
        joined = filterAndJoinDataframes(ic50, cellLine, filterField, drug)
        X, y = prepareFeatureAndLabelArrays(joined, nFeatures, classification)

        # 5 fold cross validation
        y_test, y_predict = kFoldCrossValidation(X, y, 5, classification)
        drugResponseDict[drug] = {'test': y_test, 'predict': y_predict}
    return drugResponseDict


def writePythonObjectToFile(object, filename):
    print('writing object to file...')
    pkl.dump(object, open(filename + ".pkl", "wb"))
    print('done writing to file...')


def loadPythonObjectFromFile(filename):
    print('loading object from file...')
    object = pkl.load( open(filename + ".pkl", "rb"))
    print('done loading from file...')
    return object


if __name__ == '__main__':
    #ic50, cellLine = createDrugResponseDataframes('v17_fitted_dose_response.csv',
                                                 # 'Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv')

    # add loop here over all drug ids...
    # CLASSIFICATION
    #drugResponseDict = createDrugResponseDict(ic50=ic50, cellLine=cellLine,
                                              #filterField='DRUG_ID', nFeatures=3, classification=True)
    # save to file
    #writePythonObjectToFile(drugResponseDict, 'Drug_Classif_Predictions')

    data = loadPythonObjectFromFile('Drug_Classif_Predictions')

    evaluatePerformance(data[1]['test'], data[1]['predict'])

    #joined = filterAndJoinDataframes(ic50, cellLine, 'DRUG_ID', 1)

    #X, y = prepareFeatureAndLabelArrays(joined)

    #y_test, y_predict = kFoldCrossValidation(X, y, 5)
    #evaluatePerformance(y_test=y_test, y_predict=y_predict)
    #for i, result in enumerate(results):
    #    print('Run ', i,'\nAccuracy: ', result[1], '\nF1 Score: ', result[2])


