import pandas as pd
from pandas.io.json import json_normalize
import os
import json as js
# needed to build all combinations of list elements
import itertools

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix# n = 182972 (rows)
# cols = 11

# one case_id = one patient's case?

# case_id amounts might be an indicator for how many snP's occur for each primary site

# remember! adjust to OUT.JSON when done! (below)


def loadData(filename):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)) as file:
        jsondata = js.load(file)

        # normalize to flatten different levels of hits
        df = json_normalize(jsondata['data']['hits'])
    return df


def loadAndModifyData(filename):
    # load data into frame
    df = loadData(filename)

    # drop unnecessary variables
    # ssm.mutation_type (always same value)
    df.drop(['case.project.project_id', 'ssm.mutation_type', 'ssm.mutation_subtype'], axis=1, inplace=True)

    # how many of the elements in list for ssm.consequence are the same?

    # IS THERE ANY POINT OF KEEPING THESE ATTRIBUTES?
    # note: ssm.consequence values /genes might be different in numbers, but always only multiples of the same gene
    df['ssm.consequence'] = df['ssm.consequence'].apply(flattenNestedJson)
    df['ssm.consequence'] = df['ssm.consequence'].apply(lambda x: x[0])
    # df['ssm.consequence.count'] = df['ssm.consequence'].apply(len)
    # df['ssm.consequence.different'] = df['ssm.consequence'].apply(lambda x: len(set(x)))

    # breaking down ssm.genomic_dna_change into multiple fields

    # chromosome position of gene
    # df['ssm.genomic_dna_change.chromosome_gene_pos'] = df['ssm.genomic_dna_change'].apply(lambda x: str(x).split('.')[1][:-3])

    # gene nucleotide change from
    # df['ssm.genomic_dna_change.nucleotide_change_from'] = df['ssm.genomic_dna_change'].apply(lambda x: str(x).split('.')[1][-3:].split('>')[0])
    # gene nucleotide change to
    # df['ssm.genomic_dna_change.nucleotide_change_to'] = df['ssm.genomic_dna_change'].apply(lambda x: str(x).split('.')[1][-3:].split('>')[1])

    # gene nucleotide change

    df['ssm.genomic_dna_change.nucleotide_change'] = df['ssm.genomic_dna_change'].apply(lambda x: str(x).split('.')[1][-3:])
    #print(df.head())
    #print(df.describe())
    #print(df.info())
    #unique = df['ssm.consequence'].unique()
    #print('unique genes:')

    #print(unique)

    #print('#unique genes:')

    #unique = df['case.primary_site'].unique()
    #print('unique primary sites:')

    #print(unique)

    #print(df['ssm.consequence'].nunique())
    # count unique values of case.case_id

    #casecount_df = df.groupby('case.case_id')['id'].nunique()

    # sort descending
    #casecount_df = casecount_df.sort_values(ascending=False)

    #print(casecount_df)


    #df = createFeatureFrame3(df)
    #save_as_csv(filename,df)
    #print(df.info())
def print_cases_per_site(filename):
    df = loadData(filename)

    # print how many cases there are per primary site
    cases_per_site = df.groupby(['case.case_id', 'case.primary_site'], as_index=True)['id'].nunique().reset_index()
    cases_per_site = cases_per_site.groupby(['case.primary_site'])['case.case_id'].nunique().reset_index().sort_values(
        ascending=False, by='case.case_id')
    print(cases_per_site)




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

    #dataframe.apply(lambda x: 1, axis=1)

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


## GENE BASED
##
def createFeatureFrame3(dataframe):
    # determine different chromosomes
    genes = dataframe['ssm.consequence'].unique()
    print(genes)
    # create columns with 0 entries
    i = 0
    #for gene in genes:
     #   dataframe[gene] = 0
      #  print('generated col' + str(i))
       # i = i + 1

    df = dataframe.apply(
        lambda x: createOrFillColumnBinaryIndex2(x, x['ssm.consequence']), axis=1)
    print('LAMBDA DONE')
    df.drop(['ssm.start_position', 'ssm.end_position'], axis=1, inplace=True)
    print('Dropping DONE')
    # sum() over all features except for snp count
    df1 = df.groupby(['case.case_id', 'case.primary_site'], as_index=False).sum()
    print(df1)

    # nunique over id for snp count
    # left out primary_site here, since relation between case_id and primary_site is 1:1
    df2 = df.groupby(['case.case_id', 'case.primary_site'], as_index=False)['id'].nunique()
    # print(df2.info())
    # concat dataframes:

    df = pd.concat([df2, df1], axis=1)

    df = df.sort_values(ascending=False, by='id')

    print(df)
    return df


def createFeatureArray(filename):
    print('loading dataframe from file...')
    df = loadData(filename)
    print('starting "preprocessing" of gene field...')
    #TODO redundand preprocessing with LoadAndModify
    df['ssm.consequence'] = df['ssm.consequence'].apply(flattenNestedJson)
    df['ssm.consequence'] = df['ssm.consequence'].apply(lambda x: x[0])

    print('grouping dataframe by case_id...')
    #starting point: data frame that only holds the case id, the primary site and id as a count
    df_case = df.groupby(['case.case_id', 'case.primary_site'], as_index=True)['id'].count().reset_index()
    print(df_case)
    #cases_array = np.array(df_case)
    print('find unique genes in list...')
    # find unique genes in dataframe
    genes = list(df['ssm.consequence'].unique())
    print(genes)
    print('different genes: ' + str(len(genes)))

    print('generating empty np array...')
    geneFeatureArray = np.zeros([df_case.shape[0],len(genes)], dtype=np.uint8)
    print('gene feature shape: ' + str(geneFeatureArray.shape))
    print('gene feature array datatype:' + str(geneFeatureArray.dtype))


    # iterate over each case and sum up gene mutation occurences
    # nested for loops?...
    print('start populating array....oh god...')
    print(df_case['case.case_id'])
    for row, case in df_case['case.case_id'].iteritems():
        filter_df = df.loc[df['case.case_id'] == str(case)]
        for i, mutation in filter_df['ssm.consequence'].iteritems():
            #print(type(genes))
            column = genes.index(mutation)
            # increment
            print(str(row), str(column) + 'incremented by 1...')
            geneFeatureArray[row, column] += 1

    print(geneFeatureArray)
    np.savetxt("foobig.csv", geneFeatureArray, delimiter=",", fmt='%.0i')
    print(df[['case.case_id','ssm.consequence']])
    # trial
    #print(genes[0], df['ssm.consequence'].loc[df['case.case_id'] == df_case['case.case_id'][78]])

#def findGeneMutations(data_frame, case_id):


    #df.loc[df['column_name'] == some_value]
def findIndexOfListElement(list, element):

    return list.index(element)

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


def createCompressedJSON(filename):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),filename)) as file:
        jsondata = js.load(file)
        compressedhits = []
        for i in range(1, len(jsondata['data']['hits'])):
            if i % 1000 == 0:
                compressedhits.append(jsondata['data']['hits'][i])
        # override hits
        jsondata['data']['hits'] = compressedhits
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "reduced_" + filename), "w") as out:
            js.dump(jsondata, out)


def save_as_csv(file_name, dataframe):
    dataframe.to_csv(file_name + '.csv', encoding='utf-8')


def load_from_csv(file_name):
    return pd.read_csv(file_name + '.csv', index_col=0, header=0)

################################################
##                                            ##
##          Random Forest Classifier          ##
##                                            ##
################################################

def RandomForestClassif(df, label_col):

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
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()



def test_numpy_array():
    print('starting')
    array = np.zeros([9000,18000], np.uint8)
    print('array done')
    print(array[900,4000])

if __name__ == "__main__":
    createFeatureArray('out.json')
    # test_numpy_array()
    #print_cases_per_site('out.json')
    # loadAndModifyData('out.json')
    # test()
    # createCompressedJSON('out.json')

    ##test forest:
    #df = load_from_csv('out_chromosome_mutation/out.json_chromosome')
    #RandomForestClassif(df, 'case.primary_site')
