# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data pertaining to direct marketing campaigns of a portuguese banking institution. The information regardind the dataset can be found here https://archive.ics.uci.edu/ml/datasets/bank+marketing

The marketing campaigns were based on phone calls to convince potential clients to subscribe to bank's term deposit. We seek to predict whether the potential client would accept and make a term deposit at the bank or not.

The best performing model found using AutoML was a Voting Ensemble with 91.6% accuracy, while the accuarcy of Logistic classifier implemented using hyperdrive was 90.7%

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

***Evaluate the logistic regression baseline.***

* The data is accessed from url - "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv", and a Tabular Dataset is created using TabularDatasetFactory 

* Data is then cleaned and pre-processed using clean_data function, this includes Binary encoding and One Hot Encoding of categorical features

* Data is then split into 80:20 ratio for training and testing

* Define a Scikit-learn based Logistic Regression model and set up a parameter sampler. We define 2 hyperparameters to be tuned, namely C and max_iter. C represents the inverse regularization parameter and max_iter represents the maximum number of iterations. Default parameters (1 for C) and (100) for max iterations gave us 0.9116 of accuracy


* HyperDrive configuration is created using a SKLearn estimator and parameter sampler

* Accuracy is calculated on the test set for each run and the best model is saved

**What are the benefits of the parameter sampler you chose?**
