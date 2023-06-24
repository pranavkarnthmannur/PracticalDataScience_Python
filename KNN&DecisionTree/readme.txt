Project:
Data Modelling and Presentation: Heart Failure Clinical Records Dataset
Classification: k-nearest neighbours and decision tree algorithm

Author:
Pranav Karnth Mannur
s3828461
s3828461@student.rmit.edu.au

Abstract: 
To find out what factors influence the death event and predicting the death event using the heart failure clinical records dataset which entails the medical records of 299 patients who had heart failure, collected during their follow up period, where each patient profile has 13 clinical features. Classification algorithms such as k-nearest neighbors and decision tree implemented to achieve the predictions.

Dataset:
The dataset used in this project is the Heart Failure Clinical Records Dataset, it entails medical records of 299 patients who had a heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.
Download: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

Technologies:
Python
Jupyter, Pandas
Machine Learning

Models:
k-nearest neighbors classification
decision tree algorithm


Requirements:
Run the following code to import all the required packages and libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

Getting started:

Dataset:
-Upload the downloaded dataset
-Read the dataset using pandas read_csv

Data Cleaning:
-Check for duplicates and drop if any 
-Check for NaN and drop if any 
-Round the value for age as decimals will not provide us proper statistics
-Check datatypes of all columns and change the age column from float to int. 

Data Exploration:

- view the decription of the data to view all the statistical details of the dataframe
- plot heatmap for all the columns to evaluate correlation
- plot a pairplot to explore the relationship between all pair of columns

	Exploring each column in the dataset:
	- Using the boxplot to view the max, min, quartiles, median of each column
	- Using the bar plot to derive the count of the categorical data
	- Exploring each column in the dataset using the respective plots

	Exploring pairs of columns in the dataset:
	- Seaborn library to plot a barplot, countplot, catplot
	- Plotting a piechart using pandas
	- Evaluate the relationship between each pair of columns 

Data Modelling: 

The death event column chosen as the target variable.
The rest of the columns used to train the model. 
20% of data used for testing and 80% for training.

Accuracy is calculated for both the models and compared.
Classification report is analyzed for both the models to compare results.

1. k-nearest neighbors classification: 
-Choosing the right value for k:
>Testing all possibible values for k between 1 and 11 and plotting a graph for the training and testing accuracy 
>Selecting a k value where the training dataset accuracy and the testing dataset accuracy is the closest.

-Choosing the value of p:
> p=1 is the Manhattan distance and p=2 is the Euclidean distance.
> Checking the accuracy value of the model in both cases and choosing the better scenario.

-Setting all the other parameters of the classifier as default

2. Decision Tree algorithm classification:
-Choosing the right criterion: gini/entropy:
>Testing both gini and entropy criterion in the decision tree algorithm 
>Selecting the scenario with higher accuracy

-Visualising the decision tree using graphviz for both gini and entropy.


Results:

As per our results: 
The accuracy score of k-nearest neighbors classification algorithm when p = 1 is 0.6666666666666666
The accuracy score of k-nearest neighbors classification algorithm when p = 2 is 0.65
The accuracy score of decision tree classification algorithm when criterion:gini is 0.75
The accuracy score of decision tree classification algorithm when criterion:entropy is 0.85







	 











