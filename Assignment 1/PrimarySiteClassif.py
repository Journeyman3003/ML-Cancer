import pandas as pd
from pandas.io.json import json_normalize
import os
import json as js
import itertools

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from collections import Counter


def createCompressedJSON(filename, n):
    """
    Helper function to create a smaller dataframe out of the original one that can be easier handled
    No real value for prediction or accuracy, just to evaluate correctness of processing
    currently, just every nth (see param) is taken and all other ones are dropped
    :param filename: the file name that you want to compress
    :param n: the n to use
    :return: none, just saves the file as "reduced_" + filename
    """
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),filename)) as file:
        jsondata = js.load(file)
        compressedhits = []
        for i in range(1, len(jsondata['data']['hits'])):
            if i % n == 0:
                compressedhits.append(jsondata['data']['hits'][i])
        # override hits
        jsondata['data']['hits'] = compressedhits
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "reduced_" + filename), "w") as out:
            js.dump(jsondata, out)


#######################################################
# some analytical functions to be called individually #
#######################################################


def printDataFrameAnalysis(df):
    """
    show a basic summary of a dataframe
    :param df:
    :return:
    """
    print(df.head())
    print(df.describe())
    print(df.info())


def print_cases_per_site(filename):
    """
    create a dataframe that shows how many cases are there per primary site
    :param filename:
    :return:
    """

    df = loadData(filename)

    # print how many cases there are per primary site
    # first step somehow obsolete...
    cases_per_site = df.groupby(['case.case_id', 'case.primary_site'], as_index=True)['id'].nunique().reset_index()
    cases_per_site = cases_per_site.groupby(['case.primary_site'])['case.case_id'].nunique().reset_index().sort_values(
        ascending=False, by='case.case_id')
    print(cases_per_site)
    return cases_per_site


def loadData(filename):
    """
    loads the given out.json into a data frame
    :param filename:
    :return:
    """
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)) as file:
        jsondata = js.load(file)

        # normalize to flatten different levels of hits
        df = json_normalize(jsondata['data']['hits'])
    return df


def loadAndModifyData(filename):
    """
    loads the dataframe from file and performs some preliminary modifications
    :param filename:
    :return:
    """
    # load data into frame
    df = loadData(filename)

    # drop unnecessary variables/cols
    df.drop(['case.project.project_id', 'ssm.mutation_type', 'ssm.mutation_subtype'], axis=1, inplace=True)

    # reduce value of ssm.consequence (array) to just string (the gene)
    # flatten nested JSON function returns a list of the "leaf-elements" of the json, in this case the gene names
    # since they are essentially the same (for each record/mutation), just take the first element in the list
    df['ssm.consequence'] = df['ssm.consequence'].apply(flattenNestedJson)
    df['ssm.consequence'] = df['ssm.consequence'].apply(lambda x: x[0])

    # the nucleotide change, e.g. "A>C"
    df['ssm.genomic_dna_change.nucleotide_change'] = df['ssm.genomic_dna_change'].apply(lambda x: str(x).split('.')[1][-3:])

    return df


def createFeatureAndLabelArray(filename):
    print('loading dataframe from file...')
    df = loadAndModifyData(filename)

    print('grouping dataframe by case_id...')
    #starting point: data frame that only holds the case id, the primary site and id as a count
    df_case = df.groupby(['case.case_id', 'case.primary_site'], as_index=True)['id'].count().reset_index()
    #print(df_case)



    #########
    # GENES #
    #########

    print('find unique genes in list...')

    # find unique genes in dataframe
    genes = list(df['ssm.consequence'].unique())
    #print(genes)
    #print('different genes: ' + str(len(genes)))

    print('generating empty np array...')
    geneFeatureArray = np.zeros([df_case.shape[0],len(genes)], dtype=np.uint8)


    # iterate over each case and sum up gene mutation occurences
    # nested for loops?...
    print('start populating array....oh god...')
    #print(df_case['case.case_id'])
    for row, case in df_case['case.case_id'].iteritems():
        filter_df = df.loc[df['case.case_id'] == str(case)]
        for i, mutation in filter_df['ssm.consequence'].iteritems():
            column = genes.index(mutation)
            # increment
            print(str(row), str(column), ' -> incremented by 1...')
            geneFeatureArray[row, column] += 1

    # validate that fields have been filled correctly:
    # for i in range(geneFeatureArray.shape[0]):
    #    for j in range(geneFeatureArray.shape[1]):
    #        print(i, j, int(df.loc[(df['case.case_id'] == df_case['case.case_id'][i]) & (
    #        df['ssm.consequence'] == genes[j])].shape[0]) == geneFeatureArray[i, j])


    ##################
    # MUTATION COUNT #
    ##################

    counts = np.array(df_case['id'], dtype=np.uint16)
    mutationCountArray = np.reshape(counts, (len(counts),1))

    #########################
    # CHROMOSOME x MUTATION #
    #########################

    # determine different chromosomes
    chromosomes = df['ssm.chromosome'].unique()

    # determine different mutations

    mutations = df['ssm.genomic_dna_change.nucleotide_change'].unique()

    chromosomeMutationCombinations = list(itertools.product(chromosomes, mutations))
    comboArray = np.zeros([df_case.shape[0], len(chromosomeMutationCombinations)], dtype=np.uint8)

    print(chromosomeMutationCombinations.index(tuple(["chr16", "C>A"])))
    for row, case in df_case['case.case_id'].iteritems():
        filter_df = df.loc[df['case.case_id'] == str(case)]
        for i, record in filter_df.iterrows():
            pair = record['ssm.chromosome'], record['ssm.genomic_dna_change.nucleotide_change']

            column = chromosomeMutationCombinations.index(pair)

            print(str(row), str(column), ' -> incremented by 1...')
            comboArray[row, column] += 1

    # validate that fields have been filled correctly:
    # for i in range(comboArray.shape[0]):
    #     for j in range(comboArray.shape[1]):
    #         print(i, j, int(df.loc[(df['case.case_id'] == df_case['case.case_id'][i]) &
    #                                (df['ssm.chromosome'] == chromosomeMutationCombinations[j][0]) &
    #                                (df['ssm.genomic_dna_change.nucleotide_change'] == chromosomeMutationCombinations[j][1])
    #                               ].shape[0]) == comboArray[i, j])

    featureArray = np.concatenate((mutationCountArray, geneFeatureArray, comboArray), axis=1)

    #np.savetxt("foobig.csv", geneFeatureArray, delimiter=",", fmt='%.0i')

    # drop unnecessary columns in label df
    siteLabelArray = np.array(df_case['case.primary_site'])

    print('Dimensions of Feature Array are: ', featureArray.shape)
    print('datatype of feature array is:', featureArray.dtype)
    #print(featureArray)

    print('corresponding dimensions of Label Array are: ', siteLabelArray.shape)
    print('datatype of label array is:', siteLabelArray.dtype)
    #print(siteLabelArray)

    genes = [(i + 1, x) for i, x in enumerate(genes)]
    chrMut = [(i + len(genes) + 1, x) for i, x in enumerate(chromosomeMutationCombinations)]

    return featureArray, siteLabelArray, [(0,"MutationCount")], genes, chrMut


def onlyMutationCountFeatureArray(filename):
    """
    auxilliary function to rapidly only create a feature array based of the mutation count
    :param filename:
    :return:
    """
    print('loading dataframe from file...')
    df = loadAndModifyData(filename)

    print('grouping dataframe by case_id...')
    # starting point: data frame that only holds the case id, the primary site and id as a count
    df_case = df.groupby(['case.case_id', 'case.primary_site'], as_index=True)['id'].count().reset_index()
    # print(df_case)

    X = np.array(df_case['id']).reshape(-1, 1)
    y = np.array(df_case['case.primary_site'])

    return X, y


def flattenNestedJson(nestedJson):
    """
    helper function to overcome the fact that JSON lists cannot really be "flattened" to dataframe columns
    little redundant functionality, initially would output a list of all the genes in "ssm.consequence" to check whether
    they are all equal
    As of now, it just creates the list temporarily and then returns the first element
    :param nestedJson:
    :return:
    """
    resultList = []
    if type(nestedJson) is dict:
        for k, v in nestedJson.items():
            resultList.extend(flattenNestedJson(v))
    elif type(nestedJson) is list:
        for v in nestedJson:
            resultList.extend(flattenNestedJson(v))
    else:
        resultList.append(nestedJson)
    return resultList


############################
#                          #
# Random Forest Classifier #
#                          #
############################


def RandomForestClassif(X, y):
    """
    500 tree random forest that runs on all available cores... only 2 :-( for me
    :param X: feature array
    :param y: label array
    :return: the trained classifier
    """

    # 500 trees, all cores
    forestClassif = RandomForestClassifier(n_estimators=500, n_jobs=-1)

    forestClassif.fit(X, y)

    return forestClassif


def visualizeForestResults(importances, std, indices, featurePoolSize, saveSuffix):
    """
    creates the feature importance plot
    :param importances: array of importances of features
    :param std: array of standard deviations associated with the feature importances
    :param indices: ordered array holding all features
    :param featurePoolSize: amount of features used
    :param saveSuffix: used to save different runs to a different file
    :return:
    """

    # test the model given the test data

    #importances = forest.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    #indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(featurePoolSize):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # min width = 1 inch, max width = 20 inch
    if featurePoolSize <= 1:
        width = 2
    elif featurePoolSize > 100:
        width = 20
    else:
        width = featurePoolSize/5
    # Plot the feature importances of the forest
    print("width of the feature plot:", width)
    plt.figure(figsize=(width, 8))
    plt.title("Feature importances")
    # plt.bar(range(featurePoolSize), importances[indices], color="r",  yerr=std[indices], align="center")
    plt.bar(range(featurePoolSize), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(featurePoolSize), range(featurePoolSize), rotation='vertical')
    plt.yticks(rotation='vertical')
    plt.xlim([-1, featurePoolSize])
    #plt.tight_layout()
    plt.savefig('Features_' + saveSuffix + '.png', format='png', bbox_inches='tight', dpi=100)
    #plt.show()
    plt.close()


def confusionMatrix(y_test, test_predictions, sortedLabels, saveSuffix):
    """
    calculate and, more importantly, plot the confusion matrix
    :param y_test: test labels
    :param test_predictions: predictions fro X_test
    :param sortedLabels: list of labels used for the axes' labels
    :param saveSuffix: used to save different runs to a different file
    :return:
    """
    confusion = confusion_matrix(y_test, test_predictions, labels=sortedLabels)
    print(confusion)

    row_sums = np.sum(confusion, axis=1)
    norm_conf = (confusion.T / row_sums).T
    fig = plt.figure(figsize=(20,20))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = confusion.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(confusion[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), sortedLabels[:width], rotation=90)
    plt.yticks(range(height), sortedLabels[:height])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix of the forest\nfor primary sites of cancer')

    plt.savefig('confusion_matrix_' + saveSuffix + '.png', format='png', bbox_inches='tight', dpi=100)

def reduce_features(X, forest, nFeatures, mutCountLabel, geneFeatureLabels, chrMutFeaturelabels):
    """
    helper function to reduce a feature array to the n most important features
    This is where it got ugly... since i worked with numpy arrays, I had to remember the features in an external list
    that corresponded to the feature indexes in the feature array X
    :param X: original feature array
    :param forest: the trained classifier (that holds the importances)
    :param nFeatures: amount of features to choose
    :param mutCountLabel: a list containing all the values (essentially at most only ONE label) for mutation counts per case
    :param geneFeatureLabels: a list containing all the values of genes used
    :param chrMutFeaturelabels: a list containing all the values for chromosome features
    :return: the new_X (feature array
    """
    importances = forest.feature_importances_

    # ordered list of indices to determine which features are the n "best"
    indices = np.argsort(importances)[::-1]

    newMutCountFeature = []
    newGeneFeatures = []
    newChrMutFeatures = []

    # uint16 should be sufficient
    # poplate with zeros initially
    newX = np.zeros([X.shape[0], nFeatures], dtype=np.uint16)

    # beware!!!: static structure, works only the first time of reduction, second time will be wrong!
    for i, index in enumerate(indices[0:nFeatures]):
        if (index == 0):
            newMutCountFeature.append((i, getTupleValueOutOfList(mutCountLabel, index)))
        elif (index >= 1) & (index <= len(geneFeatureLabels)):
            newGeneFeatures.append((i, getTupleValueOutOfList(geneFeatureLabels, index)))
        else:
            newChrMutFeatures.append((i, getTupleValueOutOfList(chrMutFeaturelabels, index)))
        newX[:, i] = X[:, index]

    return newX, newMutCountFeature, newGeneFeatures, newChrMutFeatures


def getTupleValueOutOfList(listOfTuples, tupleIndex):
    seekValue = [i[1] for i in listOfTuples if i[0] == tupleIndex]
    # "should" be safe since there is only one unique index...
    return seekValue[0]


def kFoldCrossValidation(X, y, k):
    """
    cross validation with k split points
    :param X: feature array
    :param y: label array
    :param k: number of split points
    :return:
    """
    kFold = KFold(n_splits=k)

    confidenceList = []
    importancesList = []
    stdList = []

    # needed for total confusion matrix
    y_testList = []
    predictionsList = []

    # 10% tests
    for train_index, test_index in kFold.split(X):
        print('new cross validation step started for run with', i, 'features')
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        forest = RandomForestClassif(X_train, y_train)
        test_predictions = forest.predict(X_test)

        confidenceList.append(forest.score(X_test, y_test))
        importancesList.append(forest.feature_importances_)
        stdList.append(np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0))

        y_testList.extend(y_test)
        predictionsList.extend(test_predictions)
        #print(y_testList)
        #print(predictionsList)

    # compute the average over all lists created (element-wise)
    confidence = sum(confidenceList) / len(confidenceList)
    importancesList = np.array([sum(e) / len(e) for e in zip(*importancesList)])
    stdList = np.array([sum(e) / len(e) for e in zip(*stdList)])

    indicesList = np.argsort(importancesList)[::-1]

    print('confidence level of prediction is:', confidence)

    return confidence, importancesList, stdList, indicesList, y_testList, predictionsList


if __name__ == "__main__":

    # optional: data analysis
    # df = loadAndModifyData('out.json')
    # print(df['case.case_id'].nunique())
    # printDataFrameAnalysis(df)

    # optional: create compressed json to test functionality
    # createCompressedJSON('out.json',1000)

    # optional: get information about "label bucket sizes":
    # df = print_cases_per_site('out.json')
    # df.plot(kind='bar',x=df['case.primary_site'],figsize=(8,6))
    # plt.xlabel('Primary sites')
    # plt.ylabel('Cases per primary site in the data')
    # plt.savefig('Buckets.png', format='png', bbox_inches='tight', dpi=100)
    # plt.show()
    # plt.close()


    # optional shortcut to only create the array for the feature "overall mutation count"
    # X, y = onlyMutationCountFeatureArray('reduced_out.json')

    X, y, mutCount, genes, chrMut = createFeatureAndLabelArray('out.json')
    print(mutCount)
    print(genes)
    print(chrMut)


    # 10% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    forest = RandomForestClassif(X_train, y_train)

    #retrieve a list that sorts labels by their occurence
    sortedLabels = Counter(y).items()
    sortedLabels = [x[0] for x in sortedLabels]

    #confidence = forest.score(X_test, y_test)
    test_predictions = forest.predict(X_test)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    #print('confidence level of prediction is:', confidence)

    print("Train Accuracy : ", accuracy_score(y_train, forest.predict(X_train)))
    print("Test Accuracy  : ", accuracy_score(y_test, test_predictions))

    # plot and save confusion matrix
    confusionMatrix(y_test, test_predictions, sortedLabels, '')

    # plot and save feature importances
    visualizeForestResults(importances, std, indices, X.shape[1], '')

    confidences = [accuracy_score(y_test, test_predictions)]

    # determine which amount of features should be tested
    # at this point: 1 and 10 - 100 in increments of 10
    intrange = [1]
    intrange.extend(range(10, 101, 10))

    for i in intrange:
        X_new, mutCount_new, genes_new, chrMut_new = reduce_features(X, forest, i, mutCount, genes, chrMut)

        print("\nNew run with %d features:" % i)
        print(mutCount_new)
        print(genes_new)
        print(chrMut_new)

        # k-fold cross validation, k = 10
        confidence, importancesList, stdList, indicesList, y_testTotal, predictionsTotal = kFoldCrossValidation(X_new, y, 10)

        confidences.append(confidence)
        # visualize (and save) results of feature importance
        visualizeForestResults(importancesList, stdList, indicesList, X_new.shape[1], str(i))

        confusionMatrix(y_testTotal, predictionsTotal, sortedLabels, str(i))
    [print("confidence level of run %d: %f" % (i+1, val)) for i, val in enumerate(confidences[1:])]

    # Plot the confidences with different features
    plt.figure()
    plt.title("Confidences achieved with different # features")
    plt.plot(list(map(str, intrange)), confidences[1:], '--g^')
    plt.axhline(y=confidences[0],ls='--', c='r')
    plt.savefig('confidence.pdf')
    plt.show()

    ##test forest:
    #df = load_from_csv('out_chromosome_mutation/out.json_chromosome')
    #RandomForestClassiffromCSV(df, 'case.primary_site')
