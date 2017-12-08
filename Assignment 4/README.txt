Supplementary files in this folder are some pickle deserialized python dictionaries that can be used with the given code to just visualize the results without having to re-computing these dicts

100CellLines.pkl: holds the 100 cell lines I used in the visualization
cellLinesDict_100cells_all_fitted/multiple.pkl: holds the final r2 and mse for cell lines when using the fitted outcomes (GDSC_dose_response_fitted_cosmic_mimick.csv) and multiple dose levels (GDSC_dose_response.csv) respectively

Further, I had a weird parsing bug when using the TSV Fingerprint files and loading them directly to python. However, since loading them to excel and converting them to CSV turned out to be a workaround, I added those csv files I used in this directory, too

