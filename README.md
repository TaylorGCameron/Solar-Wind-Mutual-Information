# Solar Wind Mutual Information

This package contains tools to compute and visualize pointwise mutual information quantities. Part of this code was used to generate the plots in  Cameron, T. G., Jackel, B., & Oliveira, D. M. ( 2019). Using mutual information to determine geoeffectiveness of solar wind phase fronts with different front orientations. Journal of Geophysical Research: Space Physics, 124, 1582– 1592. https://doi.org/10.1029/2018JA026080. THe package also contains code to pull and process solar wind data to generate those plots specifically.

To run just the mutual information code, numpy, scipy and matplotlib are required. ai.cdas is also required for the solar wind analysis. The package requires config.par to be located in the directory the analysis is done in. An example is included. SuperMA data files are also required. These should be year long files, one for each year, stored in a folder referenced in config.par.

To replicate the paper plots, run pull_and_process_data.py. This will pull and process ACE and SuperMAG data, and store the resulting averaged data in numpy sav files. Then, run demonstration.py to create the plots.  
