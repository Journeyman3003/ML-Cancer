Supplementary Material

-results.pkl: a deserialized python dict (using pickle) holding the results. All of the results included are already visualized in the write-up "prediction-melanoma-incident.pdf", but one could use this file to re-generate the graphs presented.

-county-adjacency.txt: holding county adjacency information (used for Task 2), downloaded from https://www.census.gov/geo/reference/county-adjacency.html. This file is needed to run the code for task 2

-median_income.csv: the income data obtained as instructed. Needed to run the code for both task 1 and 2

-NLDAS_Daily_Sunlight_1979-2011: the daily sunlight data obtained as instructed. Needed to run the code for both task 1 and 2

WHEN RUNNING THE CODE:

please make sure that all of the files above (except for results.pkl, which has to be in the same directory) are placed in a sub directory named "data". Otherwise, adjust the global "dataPath" variable in the supplied code.