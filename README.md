# FIA project

This repository contains the project that I have to submit to pass the course "Fondamemti di Intelligenza Artificiale" (Fundamentals of Artificial Intelligence) of 
my 3rd year of university 2022/2023 (Engineering in computer science).

The project consists in implementing the artificial intelligence algorithms studied in class, and then train and test them with the same set of data retrive from 
[here](https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip); this dataset consist in a .csv file that contains a table with 
61 columns where the last column is the value to estimate with the algorithms (also we skipped the first 2 column because are insignificant).

## Algorithms

### Decision Tree
A decision tree is a representation of a function that maps a vector of attribute values to a single output value: a "decision";
It reaches its decision by performing a sequence of tests, starting at the root and following the appropriate branch until a leaf is reached.
Each internal node in the tree corresponds to a test of the value of one of the input attributes and the leaf nodes specify what value is to be returned by the 
function.

In general the input and output values can be discrete or continuous, but in this implementation I have only considered the binary discrete classification: so I
use the median of an attribute to split the dataset.

The power of this algorithm is that it can easily represent every expression in disjunctive normal form (like "StateA OR StateB OR ..."); but in the 
other hand it has drawbacks like the fact that with real-valued attributes, the function `y > X1 + X2` is hard to represent with a decision tree because 
the decision boundary is a diagonal line, and all decision tree tests divide the space up into rectangular, axis-aligned boxes.

### Linear Regression
Linear regression is an algorithm that is used to predict a continuous value based on a given set of independent variables (features). 
this regression can be performed in two ways: a closed form solution and an iterative solution. The closed solution isa based on a single math formula unlike 
the iterative solution which use the Gradient Descent algorithm, that is an algorithm that is used to find the minimum of a function.
In this implementation I have used the Gradient Descent algorithm to find the parameters that minimize the Root Mean Squared Error (RMSE) between my predictions and the 
ground truth provided by the dataset.

### Logistic Regression
!!! To Be Completed !!!

### Nearest Neighbors
The nearest neighbors algorithm is a simple, easy-to-implement supervised learning algorithm that can be used to solve both classification and regression problems.

The algorithm assumes that similar things are near to each other, so the prediction is made by looking at the k nearest points in the training data and performing 
the mean (for regression) or the majority vote (for classification). By default, the distance between two points is calculated using the Euclidean distance, but it can be changed with other distance functions of the [Minkowski](https://en.wikipedia.org/wiki/Minkowski_distance) family.

The power of this algorithm is that it is very simple to implement and understand, but in the other hand it has drawbacks like the fact that it is very slow.

## Setup & Run
1) Clone the repository in your machine
    ```cmd
    git clone https://github.com/ValerioCeccarelli/fia_project
    ```
2) Get all package required
    ```cmd
    pip install -r requirements.txt
    ```
3) Download the dataset using a python script
    ```cmd
    python .\dataset\download_dataset.py .\dataset
    ```
4) Run the desired python algorithm (for example the "decision tree")
    ```cmd
    python .\decision_tree\main.py
    ```

### Compatibility

This project has been tested on Windows 10 with Python 3.9 and on Windows 11 with Python 3.10.
