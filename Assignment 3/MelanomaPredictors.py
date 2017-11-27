import pandas as pd
import os
import logging

# directory where to find data
dataPath = 'data'


def summarizeDF(df):
    print(df.head())
    print('df.shape:',df.shape)
    print(df.info())
    print(df.describe())


def loadFileToDf(fileName, separator=',', skip=0, nrows=None, header=0, encoding=None):

    filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath, fileName)
    dataFrame = pd.read_csv(filePath, sep=separator, skiprows=skip, nrows=nrows, header=header, encoding=encoding)

    return dataFrame

def mergeDataFrames(mergeColumn, mainFrame, *dataframes):

    #tempFrame = mainFrame
    for frame in dataframes:
        mainFrame = mainFrame.merge(frame[0], how='inner', left_on=mergeColumn, right_on=frame[1])

    return mainFrame


# based on file
# county_adjacency.txt
def createCountyAdjacencyFrame():
    df = loadFileToDf('county_adjacency.txt', separator='\t', header=None, encoding='iso-8859-1')
    df.columns = ['base_county', 'base_FIPS', 'adj_county', 'adj_FIPS']
    df=df.fillna(method='ffill')
    df['base_FIPS'] = df['base_FIPS'].astype(int)
    return df


if __name__ == "__main__":
    # melanomaFrame = loadFileToDf('melanoma-v08.txt', separator='\t')
    # summarizeDF(melanomaFrame)
    #
    # melanomaFrame = melanomaFrame[['County', ' FIPS', 'Age-Adjusted Incidence Rate']]
    # summarizeDF(melanomaFrame)

    # incomeFrame = loadFileToDf('median_income.csv', skip=6, nrows=3142)
    # summarizeDF(incomeFrame)

    # print(incomeFrame['County'].unique())
    #
    # 1979-2011
    sunlightFrame1 = loadFileToDf('NLDAS_Daily_Sunlight_1979-2011.txt', separator='\t', nrows=3111, encoding='iso-8859-1')
    sunlightFrame1.rename(columns={'Avg Daily Sunlight (KJ/m²)': 'Avg Daily Sunlight'}, inplace=True)
    sunlightFrame1 = sunlightFrame1[['County', 'County Code', 'Avg Daily Sunlight', 'Min Daily Sunlight', 'Max Daily Sunlight']]
    # 2000-2011
    sunlightFrame2 = loadFileToDf('NLDAS_Daily_Sunlight_2000-2011.txt', separator='\t', nrows=3111, encoding='iso-8859-1')

    summarizeDF(sunlightFrame1)
    #
    # #incomeFrame.merge(melanomaFrame, on='FIPS')
    #
    # # the space though...
    # merge = mergeDataFrames(' FIPS', melanomaFrame, (incomeFrame,' FIPS'), (sunlightFrame1,'County Code'))
    #
    # summarizeDF(merge)

