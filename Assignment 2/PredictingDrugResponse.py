import pandas as pd
import os
import logging

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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

# to deserialize/serialize the results to/from disk
import pickle as pkl

from itertools import cycle
from collections import Counter

# ADJUST/set to blank if required
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


def createRealCellLineRanking(ic50Filename, cellLineFilename, cosmicID, filterAUC=0.0):
    ic50, cellLine = createDrugResponseDataframes(ic50Filename=ic50Filename,
                                                  cellLineFilename=cellLineFilename,
                                                  filterAUC=filterAUC)

    # retrieve combined data for all the drugs that were tested on a given cell line
    joined = filterAndJoinDataframes(ic50Frame=ic50,cellLineFrame=cellLine,ic50FilterColumn='COSMIC_ID',
                                     ic50FilterValue=cosmicID)

    # drop all unnecessary columns
    joined = joined[['COSMIC_ID', 'DRUG_ID', 'LN_IC50']]

    tuples = [(row['DRUG_ID'], row['LN_IC50']) for _, row in joined.iterrows()]

    sortedRanking = sorted(tuples, key=lambda drug: drug[1])

    return sortedRanking



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


def plotAndSavePerformance(mainFile, labels, filenameList = None, classification=True, ticks=True, saveName=""):

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

    plt.title('Performance of by-drug ' + ('classification' if classification else 'regression') + ' models')
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

    plt.xlim(xmin=-3, xmax=len(xlabels) + 3)
    plt.legend(loc='upper right')

    if ticks:
        plt.xticks(range(len(xlabels)), xlabels, rotation='vertical')

        plt.gca().margins(x=0)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        m = 0.2  # inch margin
        s = maxsize / plt.gcf().dpi * len(xlabels) + 2 * m
        margin = m / plt.gcf().get_size_inches()[0]

        plt.gcf().subplots_adjust(left=margin, right=1. - margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

        # color 10 best and 10 worst

        [i.set_color("green") for i in plt.gca().get_xticklabels()[:10]]

        [i.set_color("red") for i in plt.gca().get_xticklabels()[len(xlabels)-10:]]
    else:
        plt.xticks(range(len(xlabels)), [''] * len(xlabels))


    if saveName != "":
        plt.savefig(saveName+'.png', format='png', bbox_inches='tight', dpi=100)
    #plt.show()
    plt.close()


def plotAndSaveCellLineRanking(rankingFile, cosmicID, ticks=True, saveName=''):

    cellLineRankingsDict = loadPythonObjectFromFile(rankingFile)
    cellLineRankings = cellLineRankingsDict[cosmicID]['drugPredict']

    # sorted ranking for comparison
    comparison = createRealCellLineRanking(ic50Filename='v17_fitted_dose_response.csv',
                                           cellLineFilename='Cell_line_COSMIC_ID_gene_expression_transposed_clean.tsv',
                                           cosmicID=cosmicID, filterAUC=0.8)

    x, y = zip(*comparison)

    predicted = []
    for drug in x:
        predicted.append([val for val in cellLineRankings if val[0] == drug][0])

    _, y_predicted = zip(*predicted)

    # compute spearman rank correlation
    comparisonrank = stats.rankdata(y, method='average')

    predictedrank = stats.rankdata(y_predicted, method='average')

    spearmanRCol, pvalue = stats.spearmanr(comparisonrank,predictedrank)

    print(spearmanRCol,pvalue)

    fig = plt.figure(figsize=(20, 8))

    plt.plot(range(len(x)), y_predicted, '--g^', linewidth=0.5,
             label='Corresponding predicted performance of drugs on cell line ' + str(cosmicID))
    plt.plot(range(len(x)), y, '--r.', linewidth=0.5,
             label='Performance of drugs on cell line ' + str(cosmicID) + ' in descending order')

    ax = fig.add_subplot(111)
    plt.text(0.02, 0.8, 'Spearman Rank Correlation: ' + str(round(spearmanRCol, 4)) +'\np-value: ' + format(pvalue, '.3g'),
             horizontalalignment='left',
             verticalalignment='center',
             transform=ax.transAxes,
             bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

    if ticks:
        plt.xticks(range(len(x)), x, rotation='vertical')

        plt.gca().margins(x=0)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        m = 0.2  # inch margin
        s = maxsize / plt.gcf().dpi * len(x) + 2 * m
        margin = m / plt.gcf().get_size_inches()[0]

        plt.gcf().subplots_adjust(left=margin, right=1. - margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    else:
        plt.xticks(range(len(x)), [''] * len(x))

    plt.xlabel('Drug ID')
    plt.ylabel('ln(IC50) for each Drug')
    plt.xlim(xmin=-3, xmax=len(x)+3)
    plt.legend(loc='upper left')

    if saveName != "":
        plt.savefig(saveName+'_Comparison.png', format='png', bbox_inches='tight', dpi=100)
    #plt.show()
    plt.close()

    # show only performance of tree

    desc = sorted(cellLineRankings, key=lambda drug: drug[1])

    x_predicted, y_predicted = zip(*desc)

    plt.figure(figsize=(20, 8))

    plt.plot(range(len(x_predicted)), y_predicted, '--g^', linewidth=0.5,
             label='Predicted performance of drugs on cell line ' + str(cosmicID) + ' in descending order')

    plt.xlabel('Drug ID')
    plt.ylabel('ln(IC50) for each Drug')
    plt.xlim(xmin=-3, xmax=len(x_predicted) + 3)

    if ticks:
        plt.xticks(range(len(x_predicted)), x_predicted, rotation='vertical')

        plt.gca().margins(x=0)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max([t.get_window_extent().width for t in tl])
        m = 0.2  # inch margin
        s = maxsize / plt.gcf().dpi * len(x_predicted) + 2 * m
        margin = m / plt.gcf().get_size_inches()[0]

        plt.gcf().subplots_adjust(left=margin, right=1. - margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

        # color 10 best and 10 worst

        [i.set_color("green") for i in plt.gca().get_xticklabels()[:10]]

        [i.set_color("red") for i in plt.gca().get_xticklabels()[len(x_predicted) - 10:]]

    else:
        plt.xticks(range(len(x_predicted)), [''] * len(x_predicted))

    plt.legend(loc='upper left')

    if saveName != "":
        plt.savefig(saveName+'.png', format='png', bbox_inches='tight', dpi=100)
    #plt.show()
    plt.close()


if __name__ == '__main__':

    # yes, deactivating warnings might sometimes be a bad idea, but this time it should be fine...
    pd.options.mode.chained_assignment = None

    np.random.seed(42)

    """
    TASK 1 PLOTS
    """
    """
    CLASSIFICATION
    """
    # plotAndSavePerformance('Drug_Classif_Predictions30_80_z2', ['Performance of classification models'],
    #                        classification=True, ticks=False,
    #                        saveName='ClassificationEvaluation')
    #
    # plotAndSavePerformance('Drug_Classif_Predictions30_80_z2', ['Performance of classification models'],
    #                        classification=True, ticks=True,
    #                        saveName='ClassificationEvaluation_Ticks')
    # """
    # REGRESSION
    # """
    # plotAndSavePerformance('Drug_Regr_Predictions30_80', ['Performance of regression models'],
    #                        classification=False, ticks=False,
    #                        saveName='RegressionEvaluation')
    #
    # plotAndSavePerformance('Drug_Regr_Predictions30_80', ['Performance of regression models'],
    #                        classification=False, ticks=True,
    #                        saveName='RegressionEvaluation_Ticks')
    #
    # """
    # SMOTE-TOMEK vs. UNBALANCED DATA
    # """
    # plotAndSavePerformance('Drug_Classif_Predictions30_80_SMOTETOMEK_z1', ['SMOTE + Tomek Links', 'unbalanced data'],
    #                       ['Drug_Classif_Predictions30_80_z1'], classification=True, ticks=False,
    #                       saveName='ClassificationSMOTEComparison')
    #
    # plotAndSavePerformance('Drug_Classif_Predictions30_80_SMOTETOMEK_z1', ['SMOTE + Tomek Links', 'unbalanced data'],
    #                        ['Drug_Classif_Predictions30_80_z1'], classification=True, ticks=True,
    #                        saveName='ClassificationSMOTEComparison_Ticks')
    # """
    # TASK 2 PLOTS
    # """
    # """
    # CLASSIFICATION vs. CLASSIFICATION + ADDINFO
    # """
    # plotAndSavePerformance('Drug_Classif_Predictions30_80_z2', ['default', 'with CNV and mutation count'],
    #                        ['Drug_Classif_Predictions30_80_z2_ADDINFO'], classification=True, ticks=False,
    #                        saveName='ClassificationComparison')
    #
    # plotAndSavePerformance('Drug_Classif_Predictions30_80_z2', ['default', 'with CNV and mutation count'],
    #                        ['Drug_Classif_Predictions30_80_z2_ADDINFO'], classification=True, ticks=True,
    #                        saveName='ClassificationComparison_Ticks')
    # """
    # CLASSIFICATION vs. CLASSIFICATION + ADDINFO
    # """
    # plotAndSavePerformance('Drug_Regr_Predictions30_80',['default', 'with CNV and mutation count'],
    #                        ['Drug_Regr_Predictions30_80_ADDINFO'], classification=False, ticks=False,
    #                        saveName='RegressionComparison')
    #
    # plotAndSavePerformance('Drug_Regr_Predictions30_80', ['default', 'with CNV and mutation count'],
    #                        ['Drug_Regr_Predictions30_80_ADDINFO'], classification=False, ticks=True,
    #                        saveName='RegressionComparison_Ticks')
    """
    TASK 3 PLOTS
    """
    plotAndSaveCellLineRanking(rankingFile='CellLineRankings', cosmicID=906794, saveName='906794_Ticks', ticks=True)
    plotAndSaveCellLineRanking(rankingFile='CellLineRankings', cosmicID=906794, saveName='906794', ticks=False)

    plotAndSaveCellLineRanking(rankingFile='CellLineRankings', cosmicID=924100, saveName='924100_Ticks', ticks=True)
    plotAndSaveCellLineRanking(rankingFile='CellLineRankings', cosmicID=924100, saveName='924100', ticks=False)

    plotAndSaveCellLineRanking(rankingFile='CellLineRankings', cosmicID=683665, saveName='683665_Ticks', ticks=True)
    plotAndSaveCellLineRanking(rankingFile='CellLineRankings', cosmicID=683665, saveName='683665', ticks=False)

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


