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


def loadFileToDf(fileName, separator=',', skip=0, header=0, encoding=None):

    filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataPath, fileName)
    dataFrame = pd.read_csv(filePath, sep=separator, skiprows=skip, header=header, encoding=encoding)

    return dataFrame


if __name__ == "__main__":
    melanomaFrame = loadFileToDf('melanoma-v08.txt', separator='\t')
    summarizeDF(melanomaFrame)

    incomeFrame = loadFileToDf('median_income.csv', skip=6)
    summarizeDF(incomeFrame)

    # 1979-2011
    sunlightFrame1 = loadFileToDf('NLDAS_Daily_Sunlight_1979-2011.txt', separator='\t', encoding='iso-8859-1')
    # 2000-2011
    sunlightFrame2 = loadFileToDf('NLDAS_Daily_Sunlight_2000-2011.txt', separator='\t', encoding='iso-8859-1')

    summarizeDF(sunlightFrame2)
