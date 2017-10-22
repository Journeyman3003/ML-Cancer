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

# n = 182972 (rows)
# cols = 11

# one case_id = one patient's case?

# case_id amounts might be an indicator for how many snP's occur for each primary site

# remember! adjust to OUT.JSON when done! (below)

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


def save_as_csv(file_name, dataframe):
    dataframe.to_csv(file_name + '.csv', encoding='utf-8')


def load_from_csv(file_name):
    return pd.read_csv(file_name + '.csv', index_col=0, header=0)

########################################################################################################################
####                                some analytical functions to be called individually                             ####
########################################################################################################################


def printDataFrameAnalysis(df):
    print(df.head())
    print(df.describe())
    print(df.info())


def print_cases_per_site(filename):
    df = loadData(filename)

    # print how many cases there are per primary site
    cases_per_site = df.groupby(['case.case_id', 'case.primary_site'], as_index=True)['id'].nunique().reset_index()
    cases_per_site = cases_per_site.groupby(['case.primary_site'])['case.case_id'].nunique().reset_index().sort_values(
        ascending=False, by='case.case_id')
    print(cases_per_site)


def loadData(filename):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)) as file:
        jsondata = js.load(file)

        # normalize to flatten different levels of hits
        df = json_normalize(jsondata['data']['hits'])
    return df


def loadAndModifyData(filename):
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


def createFeatureFrame(dataframe):
    """
    WITHOUT all possible combinations of chromosome + mutation (only seen as individual features)
    :param dataframe:
    :return:
    """
    # create feature: transformation binary index of chromosomes

    # determine different chromosomes
    # chromosomes = dataframe['ssm.chromosome'].unique()

    # although redundant: change to static list
    chromosomes = list(map(str,list(range(1, 23, 1))))
    chromosomes.extend(['X', 'Y'])
    chromosomes = list(map(lambda x: 'chr' + x, chromosomes))
    # print(chromosomes)

    # start by creating column with 0 only
    for chromosome in chromosomes:
        dataframe[chromosome] = 0

    # determine columns to set to 1
    dataframe = dataframe.apply(lambda x: createOrFillColumnBinaryIndex(x, x['ssm.chromosome']), axis=1)
    print('dataframe:')
    print(dataframe)
    print(dataframe['chr2'])


def createFeatureFrame2(dataframe):
    # determine different chromosomes
    chromosomes = dataframe['ssm.chromosome'].unique()

    # determine different mutations

    mutations = dataframe['ssm.genomic_dna_change.nucleotide_change'].unique()

    combinations = list(itertools.product(chromosomes, mutations))
    # print(chromosomes)
    # print(len(chromosomes))
    # print(mutations)
    # print(len(mutations))
    # print(combinations)
    # print(len(combinations))
    # print(combinations)

    combinations = list(map(lambda x: mergeColumnNames([x[0], x[1]]), combinations))
    for combination in combinations:
        dataframe[combination] = 0
    df = dataframe.apply(lambda x: createOrFillColumnBinaryIndex2(x, x['ssm.chromosome'], x['ssm.genomic_dna_change.nucleotide_change']), axis=1)

    # fill all generated NaN fields with 0
    #df.fillna(0, inplace=True)
    print(df)

    # cut all unnecessary cols
    df.drop(['ssm.start_position', 'ssm.end_position'], axis=1, inplace=True)


    # sum() over all features except for snp count
    df1 = df.groupby(['case.case_id', 'case.primary_site'], as_index=False).sum()
    print(df1)

    # nunique over id for snp count
    # left out primary_site here, since relation between case_id and primary_site is 1:1
    df2 = df.groupby(['case.case_id', 'case.primary_site'], as_index=False)['id'].nunique()
    #print(df2.info())
    # concat dataframes:

    df = pd.concat([df2, df1], axis=1)

    df = df.sort_values(ascending=False, by='id')

    print(df)
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
    print(genes)
    print('different genes: ' + str(len(genes)))

    print('generating empty np array...')
    # +1 to include column for snp count
    # check if works for uint8?
    # has to be uint16 since the counts range to ~2000
    geneFeatureArray = np.zeros([df_case.shape[0],len(genes)], dtype=np.uint8)
    print('gene feature shape: ' + str(geneFeatureArray.shape))
    print('gene feature array datatype:' + str(geneFeatureArray.dtype))


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
    print('gene feature shape: ' + str(geneFeatureArray.shape))
    print('gene feature array datatype:' + str(geneFeatureArray.dtype))

    print(chromosomeMutationCombinations.index(tuple(["chr16", "C>A"])))
    #print(df['ssm.genomic_dna_change.nucleotide_change'].unique())
    for row, case in df_case['case.case_id'].iteritems():
        filter_df = df.loc[df['case.case_id'] == str(case)]
        #print(filter_df[['ssm.chromosome', 'ssm.genomic_dna_change.nucleotide_change']])
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

    #comboArray = np.reshape(comboArray, (len(comboArray), 1))

    featureArray = np.concatenate((mutationCountArray, geneFeatureArray, comboArray), axis=1)


    print(featureArray)
    #np.savetxt("foobig.csv", geneFeatureArray, delimiter=",", fmt='%.0i')

    # drop unnecessary columns in label df
    siteLabelArray = np.array(df_case['case.primary_site'])

    print('Dimensions of Feature Array are: ', featureArray.shape)
    print('datatype of feature array is:', featureArray.dtype)
    print(featureArray)

    print('corresponding dimensions of Label Array are: ', siteLabelArray.shape)
    print('datatype of label array is:', siteLabelArray.dtype)
    print(siteLabelArray)

    genes = [(i + 1, x) for i, x in enumerate(genes)]
    chrMut = [(i + len(genes) + 1, x) for i, x in enumerate(chromosomeMutationCombinations)]

    return featureArray, siteLabelArray, [(0,"MutationCount")], genes, chrMut


def onlyMutationCountFeatureArray(filename):
    print('loading dataframe from file...')
    df = loadAndModifyData(filename)

    print('grouping dataframe by case_id...')
    # starting point: data frame that only holds the case id, the primary site and id as a count
    df_case = df.groupby(['case.case_id', 'case.primary_site'], as_index=True)['id'].count().reset_index()
    # print(df_case)

    X = np.array(df_case['id']).reshape(-1, 1)
    y = np.array(df_case['case.primary_site'])

    return X, y


def mergeColumnNames(columnNames):
        """
        given multiple strings 'a','b','c', it returns 'a_b_c'
        :param columnNames
        :return:
        """
        result = ''
        for columnName in columnNames:
            result = result + '_' + str(columnName)
        # cut first underscore
        result = result[1:]
        return result


def createOrFillColumnBinaryIndex(x, column_name):
    x[column_name] = 1
    return x


def createOrFillColumnBinaryIndex2(x, *columnNames):
    # first: determine the column name where the binary indicator has to be set:
    featureColumnName = mergeColumnNames(list(columnNames))

    x[featureColumnName] = 1
    print(featureColumnName)
    return x


def flattenNestedJson(nestedJson):
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
    forestClassif = RandomForestClassifier(n_estimators=500)

    forestClassif.fit(X, y)

    return forestClassif


def visualizeForestResults(importances, std, indices, featurePoolSize, saveSuffix):

    # test the model given the test data

    #importances = forest.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    #indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(featurePoolSize):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    width = featurePoolSize/5 if featurePoolSize >1 else featurePoolSize
    # Plot the feature importances of the forest
    plt.figure(figsize=(width+1,8))
    plt.title("Feature importances")
    # plt.bar(range(featurePoolSize), importances[indices], color="r",  yerr=std[indices], align="center")
    plt.bar(range(featurePoolSize), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(featurePoolSize), range(featurePoolSize), rotation='vertical')
    plt.yticks(rotation='vertical')
    plt.xlim([-1, featurePoolSize])
    #plt.tight_layout()
    plt.savefig('foo' + saveSuffix + '.pdf')
    #plt.show()
    plt.close()


def reduce_features(X, forest, nFeatures, mutCountLabel, geneFeatureLabels, chrMutFeaturelabels):
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    newMutCountFeature = []
    newGeneFeatures = []
    newChrMutFeatures = []

    # uint16 should be sufficient
    newX = np.zeros([X.shape[0], nFeatures], dtype=np.uint16)

    # beware!!!: static structure, works only the first time of reduction, second time will be wrong!
    for i, index in enumerate(indices[0:nFeatures]):
        if (index == 0):
            # print("index is mutcount")
            # print("index:", index)
            # print("number of Feature:", i)
            newMutCountFeature.append((i, getTupleValueOutOfList(mutCountLabel, index)))
        elif (index >= 1) & (index <= len(geneFeatureLabels)):
            # print("index is genefeature")
            # print("index:", index)
            # print("number of Feature:", i)
            newGeneFeatures.append((i, getTupleValueOutOfList(geneFeatureLabels, index)))
        else:
            # print("index is chrMutCrossover")
            # print("index:", index)
            # print("number of Feature:", i)
            newChrMutFeatures.append((i, getTupleValueOutOfList(chrMutFeaturelabels, index)))
        newX[:, i] = X[:, index]

    return newX, newMutCountFeature, newGeneFeatures, newChrMutFeatures

def getTupleValueOutOfList(listOfTuples, tupleIndex):
    seekValue = [i[1] for i in listOfTuples if i[0] == tupleIndex]
    # "should" be safe since there is only one unique index...
    return seekValue[0]


def RandomForestClassiffromCSV(df, label_col):

    # split dataframe /now converted numpy array into features X
    # and outcomes y

    # features, drop case_id since it's Many-1 descriptive
    X = np.array(df.drop([label_col, 'case.case_id'], 1))

    print(df.drop([label_col, 'case.case_id'], 1).info())
    # outcomes
    y = np.array(df[label_col])

    # 20% train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    forest = RandomForestClassifier(n_estimators=500)

    forest.fit(X_train, y_train)

    # test the model given the test data
    confidence = forest.score(X_test, y_test)

    print('confidence level of prediction is:')
    print(confidence)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)

    plt.xlim([-1, X.shape[1]])
    plt.show()


def kFoldCrossValidation(X, y, k):
    kFold = KFold(n_splits=k)

    confidenceList = []
    importancesList = []
    stdList = []

    # 10% tests
    for train_index, test_index in kFold.split(X):
        print('new cross validation step started for run with', i, 'features')
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.1)
        forest = RandomForestClassif(X_train, y_train)
        confidenceList.append(forest.score(X_test, y_test))
        importancesList.append(forest.feature_importances_)
        stdList.append(np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0))

    # compute the average over all lists created (element-wise)
    confidence = sum(confidenceList) / len(confidenceList)
    importancesList = np.array([sum(e) / len(e) for e in zip(*importancesList)])
    stdList = np.array([sum(e) / len(e) for e in zip(*stdList)])

    indicesList = np.argsort(importancesList)[::-1]

    print('confidence level of prediction is:', confidence)

    return confidence, importancesList, stdList, indicesList


if __name__ == "__main__":

    # optional: create compressed json to test functionality
    # createCompressedJSON('out.json',1000)

    # optional: get information about "label bucket sizes":
    # print_cases_per_site('out.json')

    # X, y = onlyMutationCountFeatureArray('reduced_out.json')
    X, y, mutCount, genes, chrMut = createFeatureAndLabelArray('reduced_out.json')
    print(mutCount)
    print(genes)
    print(chrMut)

    # 10% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    forest = RandomForestClassif(X_train, y_train)

    confidence = forest.score(X_test, y_test)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print(type(indices[0]))
    print('confidence level of prediction is:')
    print(confidence)

    visualizeForestResults(importances, std, indices, X.shape[1], '')

    confidences = []

    intrange = [1]
    intrange.extend(range(10, 101, 10))

    for i in intrange:
        X_new, mutCount_new, genes_new, chrMut_new = reduce_features(X, forest, i, mutCount, genes, chrMut)

        print("\nNew run with %d features:" % i)
        print(mutCount_new)
        print(genes_new)
        print(chrMut_new)

        # k-fold cross validation, k = 10

        confidence, importancesList, stdList, indicesList = kFoldCrossValidation(X_new, y, 2)

        confidences.append(confidence)
        # visualize (and save to pdf) results of feature importance
        visualizeForestResults(importancesList, stdList, indicesList, X_new.shape[1], str(i))
    [print("confidence level of run %d: %f" % (i+1, val)) for i, val in enumerate(confidences)]

    # Plot the confidences with different features
    plt.figure()
    plt.title("Confidences achieved with different # features")
    plt.plot(list(map(str, intrange)), confidences, '--g^')

    plt.savefig('confidence.pdf')
    plt.show()

    ##test forest:
    #df = load_from_csv('out_chromosome_mutation/out.json_chromosome')
    #RandomForestClassiffromCSV(df, 'case.primary_site')
