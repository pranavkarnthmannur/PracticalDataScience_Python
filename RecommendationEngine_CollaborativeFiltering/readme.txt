Project:
A New Collaborative Filtering Approach Utilizing Item’s Popularity

Author:
Pranav Karnth Mannur
s3828461
s3828461@student.rmit.edu.au

Abstract: (From the reserach paper) 
Collaborative filtering (CF) is one of the most successful technologies in recommender systems, and widely used in many personalized recommender areas, such as ecommerce, digital library and so on. However, most collaborative filtering algorithms suffer from data sparsity which leads to inaccuracy of recommendation. In this project, we focus on nearest-neighbor CF algorithms and propose a new collaborative filtering approach. First, we suggest a new missing data making up strategy before user’s similarity computation, which smoothes the sparsity problem. Meanwhile, the notion of item’s popularity weight is defined and introduced into the computation. After then, when facing with new users, we also find a kind way to alleviatthe difficulty in recommendation. The experimental results show our proposed approach outperforms the other existing collaborative filtering algorithms. It can efficiently smooth the inaccuracy caused by ratings sparsity, and can work well in generating recommendation for new users.

Dataset:
The dataset used in this project is the Movie Lens Dataset. 
This data set consists of:
	* 100,000 ratings (1-5) from 943 users on 1682 movies. 
	* Each user has rated at least 20 movies. 
        * Simple demographic info for the users (age, gender, occupation, zip)
Download: http://www.cs.umn.edu/Research/Grouplens/

Technologies:
Python
Jupyter, Pandas, Numpy
Machine Learning
Recommendation Systems

Models:
k-nearest neighbors collaborative filtering algorithms - Recommendation System

Requirements:
Run the following code to import all the required packages and libraries:

import pandas as pd
import numpy as np

Getting started:

To Run the File:
-Start → All Programs → Anaconda3 (64-bit) → Jupyter Notebook
-Select upload
-Upload the iPYNB file to the Jupyter 
-Make sure you have loaded the dataset in the specific directory 
-Click on Cell -> Run all
-Compute and compare the results

Dataset:
-Upload the downloaded dataset
-Read the dataset using pandas read_csv

Steps:
1. Item’s Popularity Computation 
2. User’s Similarity Computation and KNN Selection 
->Ratings data making up 
->Similarity computation utilizing item’s popularity
->K-nearest neighbors’ selection 
3. Prediction and Recommendation for Active Users


Results:
Before: MAE: 0.8471711011333851, RMSE: 1.092846045041526

After: MAE: 0.7822435788080622, RMSE: 1.002498090002132