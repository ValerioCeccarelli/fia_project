# fia_project

This repository contains the project that i have to submit to pass the course "Fondamemti di Intelligenza Artificiale" (Fundamentals of Artificial Intelligence) of my 3rd year of university 2022/2023
(Engineering in computer science).

The project consists in implementing the artificial intelligence algorithms studied in class, and then train and test them with the same set of data retrive from 
[here](https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip); this dataset consist in a .csv file that contains a table with 
61 columns where the last column is the value to estimate with the algorithms (we skipped the first 2 column because are insignificant).

## Algorithm
The next sections will explain the various algorithms used.

### Decision Tree
A decision tree is a representation of a function that maps a vector of attribute values to a single output value: a "decision";
It reaches its decision by performing a sequence of tests, starting at the root and following the appropriate branch until a leaf is reached.
Each internal node in the tree corresponds to a test of the value of one of the input attributes and the leaf nodes specify what value is to be returned by the 
function.

In general the input and output values can be discrete or continuous, but in this implementation I have only considered the binary discrete classification: so I
use the median of an attribute to split the dataset.

The power of this alghoritm is that it can easily represent every expressions in disjunctive normal form (like "StateA OR StateB OR ..."); but in the 
other hand it has drawbacks like the fact that with real-valued attributes, the function `y > X1 + X2` is hard to represent with a decision tree because 
the decision boundary is a diagonal line, and all decision tree tests divide the space up into rectangular, axis-aligned boxes.

## Setup & Run
1) Clone the repository in your machine
    ```cmd
    git clone https://github.com/ValerioCeccarelli/fia_project
    ```
2) Get all package required
    ```cmd
    pip install -r requirements.txt
    ```
3) Download the dataset
    ```cmd
    python .\dataset\download_dataset.py .\dataset
    ```
   Is available a download file also for powershell (.ps1)
4) Run the desired python algorithm (for example the "decision tree")
    ```cmd
    python .\decision_tree\main.py
    ```



