# Popbots-mTurk-HITS
A code repository for the HIT code (i.e., HTML, CSS, JavaScript, jQuery) we use on mTurk to collect and QA stressful sentences. As well as the data pipeline for analyzing votes and outputing CSV files used for training and testing of predictive models used in the Popbots system.

# The following is the typical order of the data curation, processing, and classification pipeline

## 1. mTurk HITS

Explanation of HITs:
https://www.loom.com/share/226d565b3dc846fcbce164905991229b  
Copy Collection and/or QA code into the mTurk interface and copy back/push any changes.

## 2. mTurk Data Processing 
The CSV file returned from the Amazon Mechanical Turk QA is ran through a python script that assigns the stressor labels to the sentences generated from the Mechanical Turkers. It creates a CSV with analysis of the weights each stressor label had on each sentence as well as noting whether on not the sentence was a stressor and the stressor statistics (e.g. mean, standard deviation, etc of the severity of the stressor). It also creates a plot of the distribution of the amount of labels assigned to a sentence for each label. This plot can be useful for purposes of understanding the number of turker votes needed to create reliable data to use for training and testing of an algorithim.

## 3. Dataframe Filtering
This script is to be ran from the command line. The CSV file that is returned from the script in the mTurk data processing step is used as input for the filtering script. This script allows the user to change the size of the test set, the number of times the contents of the dataframe are shuffled, the sample sizes of the number of sentences within each label, whether to have the dataframe contain both/either stressor or non-stressor sentences as well as manipulate the confidence percentages for those votes, whether to have the dataframe contain both/either covid or non-covid related sentences as well as manipulate the confidence percentages for those votes, which labels to include, which sources to include, whether to keep sentences that were used as a seed sentence for inquire, and whether to distribute the test set through each of the sources rather than randomly over all sentences. 

## 4. Scikit Classification
This script is to be ran from the command line. The CSV file that is returned from the filtering script is to be used as input for the scikit classification script. This script is to be edited manually to change the algorithm settings for the classifier and then will run the classifier on the test and training set that is returned from the dataframe filtering script. It returns the classification model, the probabilities, and a text file of the classification report and confusion matrix. 

# Prerequisites 
Requires Python3 and potentially non-standard libraries including pandas, numpy, statistics, and matplotlib, Ipywidgets, pyoperators, and tkintertable that can be installed using Pip or a similar installers. For running the notebook locally, install Jupyter Notebook: https://jupyter.org/install. Browse to the local directory on command line and launch Jupyter Notebook by typing "jupyter notebook"; this should automatically open a browser window/tab (or you can browse) to http://localhost:8888/tree. When opening the notebook, ensure Python3 Kernel is running.

# BERT Pipeline
Coming soon...
