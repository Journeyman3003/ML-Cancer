import pandas as pd
import os
import logging
from collections import Counter

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# classification metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc

# regression metrics
from sklearn.metrics import r2_score


from sklearn.feature_selection import SelectKBest, chi2, f_regression

from sklearn.preprocessing import label_binarize


# SMOTE AND TOMEK
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

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

def createWESDataFrame(wesFile):
    wesFilePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), filepath, wesFile)
    # skip that unnecessary first row in csv
    wesData = pd.read_csv(wesFilePath, sep=',', header=0)

    # just extract cDNA info
    wesData = wesData[['COSMIC_ID','cDNA']]

    wesData['cDNA'] = wesData['cDNA'].apply(lambda x: str(x)[-3:])
    wesData = wesData.groupby('COSMIC_ID')['cDNA'].apply(list).reset_index()


    distinctMutations = ['A>C','A>G','A>T','C>A','C>G','C>T','G>A','G>C','G>T','T>A','T>C','T>G']

    for mutation in distinctMutations:
        wesData[mutation] = wesData['cDNA'].apply(lambda x: x.count(mutation))

    wesData.drop('cDNA', axis=1, inplace=True)
    return wesData


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


def addMutationCounter(wesFrame, featureFrame):
    return pd.merge(featureFrame, wesFrame, how='inner', on='COSMIC_ID')

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
    smote = SMOTE(k_neighbors=3, m_neighbors=10)
    tomek = TomekLinks()
    sm = SMOTETomek(smote=smote, tomek=tomek)

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


def prepareFeatureAndLabelArrays(dataFrame, nFeatures=-1, classification=True, z_value=2):

    #drop unnecessary columns
    dataFrame.drop(['COSMIC_ID', 'DRUG_ID', 'MAX_CONC_MICROMOLAR', 'AUC', 'RMSE'], axis=1, inplace=True)

    # print('\nCOMBINED LABEL AND FEATURE DATA:')
    # summarizeDF(dataFrame)

    X = np.array(dataFrame.drop('LN_IC50', axis=1))
    y = np.array(dataFrame['LN_IC50'])
    if classification:
        y = np.array(convertLabelToClassificationProblem(y, z_value=z_value))
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


def convertLabelToClassificationProblem(y, z_value):
    z_scores_y = pd.DataFrame(stats.zscore(y))
    # print('z_scores:\n', z_scores_y)
    labelArray = z_scores_y[0].apply(lambda x: determineSensitivityLabel(x, z_value))
    # print('corresponding labels:\n', labelArray)

    return labelArray


def determineSensitivityLabel(z_score, z_value):
    if z_score >= z_value:
        return "Resistant"
    elif z_score <= -z_value:
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
    plt.title('Individual and multi-class ROC')
    plt.legend(loc="lower right")
    plt.show()


def createDrugResponseDict(ic50, cellLine, filterField='DRUG_ID', nFeatures=-1, classification=True, z_value=2, smotetomek=False,
                           addInfo=False, rememberCellLines=False):

    drugResponseDict = {}
    drugTargets = None
    cnvData = None
    wesData = None
    cosmic_IDs = None
    if addInfo:
        drugTargets = createDrugTargetDataframe('Drug_targets.csv')
        print('loading copy number data and WES data (may take some time)...')
        cnvData = createCopyNumberVariationDataframe('Gene_level_CN.csv')

        wesData = createWESDataFrame('WES_variants.csv')

    for i, drug in enumerate(ic50[filterField].unique()):
        print(' #####################################\n',
              '##             RUN ',i,'             ##\n',
              '#####################################')
        joined = filterAndJoinDataframes(ic50, cellLine, filterField, drug)
        if addInfo:
            print('adding additional features, e.g. Drug targets + cell line cnvs')
            # the required file was manually assembled given the info in
            # GDSC_Screened_Compounds.csv
            # and
            # GDSC-Drugs-with-SMILES-Aliases.csv
            joined = addCopyNumberData(drug, drugTargets, cnvData, joined)

            joined = addMutationCounter(wesData, joined)

        if rememberCellLines:
            cosmic_IDs = list(joined['COSMIC_ID'])

        X, y = prepareFeatureAndLabelArrays(joined, nFeatures, classification, z_value=z_value)

        # apply smoteTomek if desired for classification problem
        if smotetomek and z_value >= 1 and classification:
            try:
                X, y = smoteTomek(X, y)
            except Exception:
                print('whoopsy daisy, you screwed up, could not apply smote tomek to drug', drug, ', skipped...')
                logging.exception("message")

        # 5 fold cross validation
        y_test, y_predict = kFoldCrossValidation(X, y, 5, classification)

        if rememberCellLines:
            drugResponseDict[drug] = {'test': y_test, 'predict': y_predict, 'COSMIC_ID': cosmic_IDs}
        else:
            drugResponseDict[drug] = {'test': y_test, 'predict': y_predict}

        print('DrugResponseDict:\n', drugResponseDict[drug])
        print('sizes of attributes:')
        print('test:', len(drugResponseDict[drug]['test']))
        print('predict:', len(drugResponseDict[drug]['predict']))
        print('COSMIC_ID:', len(drugResponseDict[drug]['COSMIC_ID']))

    return drugResponseDict

# only regression!
# def createCellLineRanking(ic50, cellLine, cosmic_ID, filterField='DRUG_ID', nFeatures=-1):
#
#     cosmicIDFeatures = cellLine[~cellLine['COSMIC_ID'].isin(cosmic_ID)]
#     if cosmicIDFeatures.shape[0] == 0:
#         print('Given cell Line with ID', cosmic_ID,
#               'cannot be analyzed since there are no gene expression values available')
#     else:
#         cellLineDict = {}
#         for i, drug in enumerate(ic50[filterField].unique()):
#             print(' #####################################\n',
#                   '##             RUN ',i,'             ##\n',
#                   '#####################################')
#             joined = filterAndJoinDataframes(ic50, cellLine, filterField, drug)
#
#             # leave out the cell line to be predicted if it is present
#             joined = joined[joined['COSMIC_ID'].isin(cosmic_ID)]
#
#             X_train, y_train = prepareFeatureAndLabelArrays(joined, nFeatures=nFeatures, classification=False)
#
#             # take test/train data out of cosmicIDFeatures
#
#             X_test = np.array(cosmicIDFeatures.drop('LN_IC50', axis=1))
#             y_test = np.array(cosmicIDFeatures['LN_IC50'])
#
#             # now, similar to leave-one-out classification, train a random forest regressor and test the cell line
#             forest = RandomForestRegr(X_train, y_train)
#
#             y_predict = forest.predict(X_test)
#
#             cellLine


def createCellLineRanking(drugResponseFile, saveName=''):
    ic50, cellLine = createDrugResponseDataframes(ic50Filename='v17_fitted_dose_response.csv',
                                                  cellLineFilename='Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
                                                  filterAUC=0.80)
    drugResponseDict = loadPythonObjectFromFile(drugResponseFile)

    cellLineRankings = {}
    for i, cell in cellLine['COSMIC_ID'].astype(int).iteritems():
        cellLineRankings[cell] = {'drugPredict': []}
        for drug in drugResponseDict:
            indexes = [j for j, cL in enumerate(drugResponseDict[drug]['COSMIC_ID']) if cL == cell]
            # account for the fact that not every cell line was screened for every drug
            if len(indexes) > 0:
                cellLineRankings[cell]['drugPredict'].append((drug, drugResponseDict[drug]['predict'][indexes[0]]))

    if saveName != "":
        writePythonObjectToFile(cellLineRankings, saveName)


    return cellLineRankings



def loadAndRun(ic50Filename, cellLineFilename, filterField, nFeatures=-1, filterAUC=0.0, classification=True, z_value=2,
               smotetomek=False, addInfo=False, rememberCellLines=False, save=True):
    ic50, cellLine = createDrugResponseDataframes(ic50Filename, cellLineFilename, filterAUC=filterAUC)
    drugResponseDict = createDrugResponseDict(ic50=ic50, cellLine=cellLine,
                                              filterField=filterField, nFeatures=nFeatures,
                                              classification=classification, z_value=z_value, smotetomek=smotetomek,
                                              addInfo=addInfo, rememberCellLines=rememberCellLines)

    # write to file if desired
    if save:
        method = 'Classif' if classification else 'Regr'
        n = str(nFeatures) if nFeatures > 0 else ''
        add = '_ADDINFO' if addInfo else ''
        z = '_z' + str(z_value) if classification else ''
        aucThreshold = '_' + str(int(filterAUC * 100))
        st = '_SMOTETOMEK' if smotetomek else ''
        cL = '_CellLines' if rememberCellLines else ''

        writePythonObjectToFile(drugResponseDict, 'Drug_'+method+'_Predictions'+ n + aucThreshold + st + z + add +
                                cL)

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


def plotAndSavePerformance(mainFile, labels, filenameList = None, classification=True, saveName=""):

    mainData = loadPythonObjectFromFile(mainFile)

    mainScore = [(str(drug), f1_score(mainData[drug]['test'], mainData[drug]['predict'], average='macro') if classification else
                          r2_score(mainData[drug]['test'], mainData[drug]['predict']))
              for drug in mainData]

    main = sorted(mainScore, key=lambda x: x[1], reverse=True)

    sortedkeys, _ = zip(*main)
    sortedkeys = list(sortedkeys)

    xlabels, y = zip(*main)

    fig = plt.figure(figsize=(20, 8))
    plt.plot(range(len(y)), y, '--g^', linewidth=0.5, label=labels[0])

    plt.title('Performance')
    plt.xlabel('Drug ID')
    if classification:
        plt.ylabel('Multiclass Macro F1 Score')
    else:
        plt.ylabel('R2 Score')

    try:
        for i, file in enumerate(filenameList):
            data = (loadPythonObjectFromFile(file))
            dataScore = [(str(drug), f1_score(data[drug]['test'], data[drug]['predict'],
                                              average='macro') if classification else
                                     r2_score(data[drug]['test'], data[drug]['predict']))
                         for drug in data]
            temp = []
            for key in sortedkeys:
                temp.append([val for val in dataScore if val[0] == key][0])

            _, y2 = zip(*temp)

            plt.plot(range(len(y)), y2, '--.', color="rbgkm"[i], linewidth=0.5, label=labels[i+1])
    except TypeError:
        print('no additional results given to plot')
        logging.exception("message")
    plt.legend(loc='upper right')
    plt.xticks(range(len(xlabels)), xlabels, rotation='vertical')
    locs, labs = plt.xticks()
    plt.xticks(locs[0:9])

    if saveName != "":
        plt.savefig(saveName+'.png', format='png', bbox_inches='tight', dpi=100)
    plt.show()


def plotAndSaveCellLineRanking(rankingFile, cosmicID, saveName=''):

    cellLineRankings = loadPythonObjectFromFile(rankingFile)
    desc = sorted(cellLineRankings[cosmicID]['drugPredict'], key=lambda drug: drug[1])

    x, y = zip(*desc)

    plt.figure(figsize=(20, 8))

    plt.plot(range(len(x)), y, '--r.', linewidth=0.5, label='Performance of drugs on cell line ' + str(cosmicID) + ' in descending order')
    plt.xticks(range(len(x)), x, rotation='vertical')
    plt.legend(loc='upper left')

    if saveName != "":
        plt.savefig(saveName+'.png', format='png', bbox_inches='tight', dpi=100)
    plt.show()

if __name__ == '__main__':

    # yes, deactivating warnings might sometimes be a bad idea, but this time it should be fine...
    pd.options.mode.chained_assignment = None

    np.random.seed(42)
    #
    # compare smote tomek to without smote tomek:
    #plotAndSavePerformance('Drug_Classif_Predictions30_80_SMOTETOMEK_z1', ['SMOTE + Tomek Links', 'unbalanced data'],
    #                       ['Drug_Classif_Predictions30_80_z1'], classification=True,
    #                       saveName='ClassificationSMOTEComparison')
    #
    #
    # plotAndSavePerformance('Drug_Classif_Predictions30_80_z2', ['default', 'with CNV and mutation count'],
    #                        ['Drug_Classif_Predictions30_80_z2_ADDINFO'], classification=True,
    #                        saveName='ClassificationComparison')
    #
    # plotAndSavePerformance('Drug_Regr_Predictions30_80',['default', 'with CNV and mutation count'],
    #                        ['Drug_Regr_Predictions30_80_ADDINFO'], classification=False,
    #                        saveName='RegressionComparison')
    #
    # # REGRESSION RUNS
    # """
    # first trial with 3 features and no AUC filter applied to ic50
    # """
    # drugResponseDict = loadAndRun(ic50Filename='v17_fitted_dose_response.csv',
    #                               cellLineFilename='Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                               filterField='DRUG_ID', nFeatures=3, filterAUC=0.0, classification=False,
    #                               addInfo=False, save=True)
    #
    # drugResponseDict = loadAndRun(ic50Filename='v17_fitted_dose_response.csv',
    #                               cellLineFilename='Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                               filterField='DRUG_ID', nFeatures=30, filterAUC=0.80, classification=False,
    #                               addInfo=False, save=True)
    #
    # drugResponseDict = loadAndRun(ic50Filename='v17_fitted_dose_response.csv',
    #                               cellLineFilename= 'Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                               filterField='DRUG_ID', nFeatures=30, filterAUC=0.80, classification=False,
    #                               addInfo=True, save=True)
    #
    # #
    # #data1 = loadPythonObjectFromFile('Drug_Regr_Predictions30_80')
    # #data2 = loadPythonObjectFromFile('Drug_Regr_Predictions30_80_ADDINFO')
    # #
    # #for key in data1:
    # #    print("Drug",key,":",r2_score(data1[key]['test'],data1[key]['predict']))
    # #    print("Drug", key, " addinfo:", r2_score(data2[key]['test'], data2[key]['predict']), "\n")
    #     #print("Drug", key, ":", mean_squared_error(data1[key]['test'], data1[key]['predict']),"\n")
    # #
    # #
    # #
    # # CLASSIFICATION RUNS (z=2)
    # """
    # first trial with 3 features and no AUC filter applied to ic50
    # """
    # drugResponseDict = loadAndRun(ic50Filename='v17_fitted_dose_response.csv',
    #                               cellLineFilename='Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                               filterField='DRUG_ID', nFeatures=3, filterAUC=0.0, classification=True, z_value=2,
    #                               smotetomek=False, addInfo=False, save=True)
    #
    # """
    # 30 features, 80% AUC filter, NO additional data/features (task 1)
    # """
    # drugResponseDict = loadAndRun(ic50Filename='v17_fitted_dose_response.csv',
    #                               cellLineFilename='Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                               filterField='DRUG_ID', nFeatures=30, filterAUC=0.80, classification=True, z_value=2,
    #                               smotetomek=False, addInfo=False, save=True)
    #
    # """
    # 30 features, 80% AUC filter, USE additional data/features (task 2)
    # """
    # drugResponseDict = loadAndRun(ic50Filename='v17_fitted_dose_response.csv',
    #                               cellLineFilename='Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                               filterField='DRUG_ID', nFeatures=30, filterAUC=0.80, classification=True, z_value=2,
    #                               smotetomek=False, addInfo=True, save=True)

    """
    SMOTE TOMEK LINK RUNS (z=1)
    """

    """
    comparison run, NO smote:
    """
    # drugResponseDict = loadAndRun(ic50Filename='v17_fitted_dose_response.csv',
    #                               cellLineFilename='Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                               filterField='DRUG_ID', nFeatures=30, filterAUC=0.80, classification=True, z_value=1,
    #                               smotetomek=False, addInfo=False, save=True)

    """
    no added info aka task 1:
    """
    # drugResponseDict = loadAndRun(ic50Filename='v17_fitted_dose_response.csv',
    #                               cellLineFilename='Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                               filterField='DRUG_ID', nFeatures=30, filterAUC=0.80, classification=True, z_value=1,
    #                               smotetomek=True, addInfo=False, save=True)

    """
    with added info aka task 2:
    """
    #drugResponseDict = loadAndRun(ic50Filename='v17_fitted_dose_response.csv',
    #                              cellLineFilename='Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                              filterField='DRUG_ID', nFeatures=30, filterAUC=0.80, classification=True, z_value=1,
    #                              smotetomek=True, addInfo=True, save=True)



    """
    TASK 3 RUNS
    """
    # drugResponseDict = loadAndRun(ic50Filename='v17_fitted_dose_response.csv',
    #                               cellLineFilename='Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
    #                               filterField='DRUG_ID', nFeatures=30, filterAUC=0.80, classification=False, save=True,
    #                               rememberCellLines=True)


    createCellLineRanking('Drug_Regr_Predictions30_80_CellLines', saveName='CellLineRankings')

    plotAndSaveCellLineRanking(rankingFile='CellLineRankings', cosmicID=906794, saveName='906794')














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
