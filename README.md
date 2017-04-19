# IRDM 2017 Group Project
## Option 2: Learning to Rank
### Group 42: I-Horng Huang, Tomas Jakab, Phillip Mortimer

---

## Pre-requisites

The Python code is written in Python 3.5.  The logistic regression and MLP implementations require the following packages:
- SciKit learn
- TensorFlow version 1.0 or greater
- Keras 2.0 or greater

The other algorithms use the Java RankLib library.

The MSLR 10k dataset should be uncompressed into a folder named 'data', with each fold in its own sub-folder named 'Fold1', 'Fold2' etc.  This is the standard format for the data if it is decompressed in the folder 'data'.

---

## How to run the code

RankLib algorithms:

The RankLib algorithms can be run using the shell files in each algorithm's named folder in the 'code' folder.  There are multiple shell files for each algorithm, used to perform the parameter search, which was run across multiple AWS instances.

Python code:

To replicate the experiments with the logistic regresion and MLP classifiers from the report, in the folder code/logistic_regression run:

`python experiments.py [EXPERIMENT_ID]`

`EXPERIMENT_ID` - ID of the experiment. Choose from the following list:
* logistic regression
  * with scikit learn
    * baseline   [1]
  * with TensorFlow baseline   [2]
    * reguralisation (8 values)   [3]
    * no feature normalization   [4]
    * class balancing   [5]
 * multilayer perceptron with feature normalization   [6]

 The code for logistic regression and MLP is implemented in the class `PointwiseRanker` in the file code/logistic_regression/pointwise_ranker.py

---

## Github folder structure

### code

There are sub-folders for each algorithm used:

- AdaRank
- LambdaMART
- RankNet
- logistic_regression

The logistic regression folder contains our own Python implementation of the logistic regression classifier, multi-layer perceptron classifer, and the NDCG and ERR metrics.

### document

This document contains the LaTeX file for the group report and images used in the figures.

### model

This folder contains saved models for the RankLib algorithms, as well as log files saved during the hyper-parameter search performed for each algorithm.

### RankLib

The .jar file for the RankLib library
