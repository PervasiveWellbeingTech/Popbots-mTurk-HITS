# Popbots-mTurk-HITS
A code repository for the HIT code (i.e., HTML, CSS, JavaScript, jQuery) we use on mTurk to collect and QA stressful sentences. As well as the data pipeline for analyzing votes and outputing CSV files used for training and testing of predictive models used in the Popbots system.

# mTurk HITS

Explanation of HITs:
https://www.loom.com/share/226d565b3dc846fcbce164905991229b  
Copy Collection and/or QA code into the mTurk interface and copy back/push any changes.

# mTurk Data Processing 
The CSV file returned from the Amazon Mechanical Turk QA is ran through a python script that assigns the stressor labels to the sentences generated from the Mechanical Turkers. It creates a CSV with analysis of the weights each stressor label had on each sentence as well as noting whether on not the sentence was a stressor and the stressor statistics (e.g. mean, standard deviation, etc of the severity of the stressor). It also creates a plot of the distribution of the amount of labels assigned to a sentence for each label. This plot can be useful for purposes of understanding the number of turker votes needed to create reliable data to use for training and testing of an algorithim. 

# BERT Pipeline
Coming soon...
