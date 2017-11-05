import pandas as pd
import os
import logging

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

# classification metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc

# regression metrics
from sklearn.metrics import r2_score, mean_squared_error


from sklearn.feature_selection import SelectKBest, chi2, f_regression

from sklearn.preprocessing import label_binarize


# SMOTE AND TOMEK
from imblearn.combine import SMOTETomek

from scipy import stats
from scipy import interp

# to deserialize/serialize the classifiers to/from disk
import pickle as pkl

from itertools import cycle
from collections import Counter

# ADJUST/set to blank
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

# input1 : GDSC_Screened_Compounds.csv
# input2: GDSC-Drugs-with-SMILES-Aliases.csv
def createDrugTargetDataframe(drugTargetFile):

    screenedFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),filepath,drugTargetFile)
    screenedDrugsData = pd.read_csv(screenedFilePath, sep=',', header=0)

    print(screenedDrugsData.info())

    screenedDrugsData['TARGET'] = screenedDrugsData['TARGET'].apply(lambda x: str(x).replace(' ', '').split(','))

    print(screenedDrugsData)

    return screenedDrugsData


def createCopyNumberVariationDataframe(copyFile):

    cnvFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),filepath, copyFile)
    # skip that unnecessary first row in csv
    cnvData = pd.read_csv(cnvFilePath, skiprows=[0], sep=',', header=0)

    # drop unnecessary cols
    cnvData.drop(cnvData.columns[[1,2,3]], inplace=True, axis=1)

    cnvData.columns.values[0] = 'GENE'
    cnvData.set_index('GENE', inplace=True)
    # transposed version: genes on columns
    cnvData = cnvData.T.reset_index()
    cnvData.columns.values[0] = 'COSMIC_ID'

    # wow that took way too long to find out that LABELS were stored as string.... and after transposing the column dtype is STILL string...
    cnvData['COSMIC_ID'] = cnvData['COSMIC_ID'].astype(int)
    cnvData = cnvData.rename_axis(None, axis=1)
    return cnvData


def addCopyNumberData(drug_id, drugTargets, cnvFrame, featureFrame):

    targetGenes = drugTargets['TARGET'].loc[drugTargets['DRUG ID'] == drug_id].reset_index()
    targetGenes = targetGenes.iloc[0]['TARGET']
    print(targetGenes)

    # warning: not all the values in array are genes, thus "protect" with try catch
    for gene in targetGenes:
        print("entry", gene, "is checked...is that even a gene?")
        try:

            cnvFrameTemp = cnvFrame[['COSMIC_ID', gene]]
            print('turns out', gene, 'is actually a gene...')
            featureFrame = pd.merge(featureFrame, transformCNVFrame(cnvFrameTemp, gene), how='inner', on='COSMIC_ID')

        except KeyError:
            print('hm, apparently', gene, 'is not a gene ¯\_(ツ)_/¯, thus it was not added...')
            #logging.exception("message")

    print(featureFrame.shape)
    return featureFrame


def transformCNVFrame(cnvFrame, gene):

    cnvFrame['Deletion'] = cnvFrame[gene].apply(lambda x: (1 if int(str(x).split(',')[1]) == 0 else 0))
    cnvFrame['PartialDeletion'] = cnvFrame[gene].apply(lambda x: (1 if int(str(x).split(',')[1]) == 1 else 0))
    cnvFrame['Normal'] = cnvFrame[gene].apply(lambda x: (1 if int(str(x).split(',')[1]) == 2 else 0))
    cnvFrame['Multiplication'] = cnvFrame[gene].apply(lambda x: (1 if int(str(x).split(',')[1]) >= 3 & int(str(x).split(',')[1]) <= 7 else 0))
    cnvFrame['DrasticMultiplication'] = cnvFrame[gene].apply(lambda x: (1 if int(str(x).split(',')[1]) >= 8 else 0))

    cnvFrame.drop(gene, axis=1, inplace=True)

    return cnvFrame


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

def smoteTomek(X, y):

    sm = SMOTETomek()

    X_resampled, y_resampled = sm.fit_sample(X,y)

    return X_resampled, y_resampled


def RandomForestRegr(X, y):
    # 500 trees, all cores
    forestRegr = RandomForestRegressor(n_estimators=500, criterion="mse", n_jobs=-1)
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
        X_new = SelectKBest(f_regression, k=nFeatures).fit_transform(X, y)
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
    computeMulticlassAuc(y_test, y_predict, ['Resistant', 'Intermediate', 'Sensitive'])

def computeMulticlassAuc(y_test, y_predict, classlabels):
    # Binarize the output
    print('Counter for test data (actual labels): ', Counter(list(y_test)))
    print('Counter for predicted data (predicted labels): ', Counter(list(y_predict)))
    y_test = label_binarize(y_test, classes=classlabels)
    y_predict = label_binarize(y_predict, classes=classlabels)
    n_classes = y_test.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i],pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predict.ravel(), pos_label=1)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw=2
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
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
        print("fpr:", classlabels[i], fpr[i])
        print("tpr:", classlabels[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classlabels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def createDrugResponseDict(ic50, cellLine, filterField, nFeatures=-1, classification=True, addInfo=False):

    drugResponseDict = {}
    drugTargets = None
    cnvData = None
    if addInfo:
        drugTargets = createDrugTargetDataframe('Drug_targets.csv')
        print('loading copy number data (may take some time)...')
        cnvData = createCopyNumberVariationDataframe('Gene_level_CN.csv')

    for i, drug in enumerate(ic50[filterField].unique()):
        print(' #####################################\n',
              '##             RUN ',i,'             ##\n',
              '#####################################')
        joined = filterAndJoinDataframes(ic50, cellLine, filterField, drug)
        if addInfo:
            print('adding additional features, e.g. Drug targets + cell line cnvs')
            # this file was manually assembled given the info in
            # GDSC_Screened_Compounds.csv
            # and
            # GDSC-Drugs-with-SMILES-Aliases.csv
            joined = addCopyNumberData(drug, drugTargets, cnvData, joined)
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

    # yes, deactivating warnings might sometimes be a bad idea, but this time it should be fine...
    pd.options.mode.chained_assignment = None

    # createDrugTargetDataframe('Drug_targets.csv')
    #df = createCopyNumberVariationDataframe('Gene_level_CN.csv')
    #
    #ic50, cellLine = createDrugResponseDataframes('v17_fitted_dose_response.csv',
    #                                              'Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                                              filterAUC=0.80)
    #
    # REGRESSION
    #drugResponseDict = createDrugResponseDict(ic50=ic50, cellLine=cellLine,
    #                                          filterField='DRUG_ID', nFeatures=30, classification=False, addInfo=True)
    # save to file
    #writePythonObjectToFile(drugResponseDict, 'Drug_Regr_Predictions30_80_ADDINFO')
    #
    data1 = loadPythonObjectFromFile('Drug_Regr_Predictions30_80')
    data2 = loadPythonObjectFromFile('Drug_Regr_Predictions30_80_ADDINFO')
    #
    for key in data1:
        print("Drug",key,":",r2_score(data1[key]['test'],data1[key]['predict']))
        print("Drug", key, " addinfo:", r2_score(data2[key]['test'], data2[key]['predict']), "\n")
        #print("Drug", key, ":", mean_squared_error(data1[key]['test'], data1[key]['predict']),"\n")
    #
    #
    #
    # CLASSIFICATION
    #drugResponseDict = createDrugResponseDict(ic50=ic50, cellLine=cellLine,
    #                                          filterField='DRUG_ID', nFeatures=30, classification=True, addInfo=True)
    # save to file
    #writePythonObjectToFile(drugResponseDict, 'Drug_Classif_Predictions30_80_z2_ADDINFO')
    #
    #data1 = loadPythonObjectFromFile('Drug_Classif_Predictions30_80_z2')
    #data2 = loadPythonObjectFromFile('Drug_Classif_Predictions30_80_z2_ADDINFO')
    #
    #evaluatePerformance(data1[1]['test'], data1[1]['predict'])
    #evaluatePerformance(data2[1]['test'], data2[1]['predict'])
    #
    # #joined = filterAndJoinDataframes(ic50, cellLine, 'DRUG_ID', 1)
    #
    # #X, y = prepareFeatureAndLabelArrays(joined)
    #
    # #y_test, y_predict = kFoldCrossValidation(X, y, 5)
    # #evaluatePerformance(y_test=y_test, y_predict=y_predict)
    #
    #
    #
    # # SMOTETomek Test
    #
    # ic50, cellLine = createDrugResponseDataframes('v17_fitted_dose_response.csv',
    #                                               'Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                                               filterAUC=0.80)
    # join = filterAndJoinDataframes(ic50,cellLine,'DRUG_ID',1)
    #
    # #BEFORE
    # X, y = prepareFeatureAndLabelArrays(join, 30, classification=True)
    # print(' Before:', '\nX:', X, '\ny:', y)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # classif = RandomForestClassif(X_train, y_train)
    #
    # evaluatePerformance(y_test, classif.predict(X_test))
    # print(Counter(y))
    #
    # #AFTER
    # X_new, y_new = smoteTomek(X,y)
    # print(' After:', '\nX:', X_new, '\ny:', y_new)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
    # classif = RandomForestClassif(X_train, y_train)
    #
    # evaluatePerformance(y_test, classif.predict(X_test))
    # print(Counter(y_new))
