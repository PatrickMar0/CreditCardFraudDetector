# main.py
# Using a data set from Kaggle, determine if a credit card transaction is fraudulent

#Imports
#Internal Libraries
import os
import pdb
#Other Files
import NN
import DT
#External Libraries (pip)
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = 'creditcard.csv'

def main():
    #Load data
    x, y = load_data()

    #Run Algorithms
    print("================================================")
    print("|    1. Press (1) for Neural Network Algorithm |")
    print("|    2. Press (2) for Decision Tree Algorithm  |")
    print("================================================")
    print("Choose the algorithm you wish to use: ")
    while True:
        userInput = int(input())

        if userInput == 1:
            NN.NN_algorithm(x, y)
            break
        if userInput == 2:
            DT.DT_algorithm(x, y)
            break
        else:
            print("Invalid Response!!! Try Again!!")
            continue
    


def load_data():
    '''Load in data from csv file and prepare it for algorithms'''
    #Grab Features
    cols = [i for i in range(1, 30)]
    features = np.genfromtxt(os.path.join(ROOT, DATA), delimiter=',', skip_header=1, usecols=cols, dtype=None)
    x = features

    #Grab Labels
    labels = np.genfromtxt(os.path.join(ROOT, DATA), delimiter=',', skip_header=1, usecols=-1, dtype=str)
    labels = np.char.replace(labels, "\"", "")
    labels = labels.astype(int)
    y = labels

    return x, y


if __name__ == "__main__":
    main()
