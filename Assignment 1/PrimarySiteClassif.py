import pandas as pd
from pandas.io.json import json_normalize
import os
import json as js

# n = 182972 (rows)
# cols = 11

# one case_id = one patient's case?

# remember! adjust to OUT.JSON when done!


def loadData(filename):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),filename)) as file:
        jsondata = js.load(file)

        # normalize to flatten different levels of hits
        df = json_normalize(jsondata['data']['hits'])
        # print(df)
    return df
        #print(list(df['ssm.mutation_type'].unique())) # only Simple somatic Mutation, drop column...
#print(os.path.dirname(os.path.realpath(__file__)))


def loadAndModifyData(filename):
    #load data into frame
    df = loadData(filename)

    # how many of the elements in list for ssm.consequence are the same?

    # IS THERE ANY POINT OF KEEPING THESE ATTRIBUTES?
    # note: ssm.consequence values /genes might be different in numbers, but always only multiples of the same gene
    df['ssm.consequence'] = df['ssm.consequence'].apply(flattenNestedJson)
    df['ssm.consequence.count'] = df['ssm.consequence'].apply(len)
    df['ssm.consequence.different'] = df['ssm.consequence'].apply(lambda x: len(set(x)))
    print(df)
    print(df.describe())
    print(df.info())

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
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"reduced_" + filename),"w") as out:
            js.dump(jsondata, out)





def test():
    json = '{"ssm": {"mutation_subtype": "Single base substitution",' \
           '"end_position": 11394531,' \
           '"start_position": 11394531,' \
           '"genomic_dna_change": "chr12:g.11394531C>A",' \
           '"mutation_type": "Simple Somatic Mutation",' \
           '"chromosome": "chr12"' \
           '},' \
           '"case": {' \
           '"project": {' \
           '"project_id": "TCGA-COAD"' \
           '},' \
           '"primary_site": "Colorectal"' \
           '},' \
           '"id": "317e2e60-8198-5620-bac8-b7e4afc6ae5d"}'
    print(json)
    obj = js.loads(json)
    norm = json_normalize(obj)
    print(norm)
    df = pd.read_json(norm)
    print(df)


if __name__ == "__main__":
    loadAndModifyData('out.json')
    # test()
    # createCompressedJSON('out.json')
