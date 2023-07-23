#DT.py
#Apply a decision Tree on a given dataset

#Imports
#Internal Libraries
import pdb
#External Libraries (pip)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

def DT_algorithm(x,y):

    print()
    print("===============================")
    print("|                             |")
    print("|   DECISION TREE ALGORITHM   |")
    print("|                             |")
    print("===============================")

    # Overfitting LMAO
    # ourTree = DecisionTreeClassifier(criterion="entropy", splitter="best")

    ourTree = DecisionTreeClassifier(
      criterion="entropy",
      splitter="best",
      max_features=17,
      max_depth=24,
      min_impurity_decrease=0.0001,
      min_samples_split=20,
      random_state=10)

    ourTree = ourTree.fit(x[:150000],y[:150000])

    plot_tree(ourTree, filled=True, rounded=True)
    print("DONE!")
    print()
    print("Testing decision tree.....")

    testResult = ourTree.predict(x[-100001:-1])


    print("DONE!")
    print()

    testLabels = y[-100001:-1]

    matrix = confusion_matrix(testResult,testLabels)
    print("==========================")
    print()
    print("Confusion Matrix: ")

    print(matrix)

    ourScore = 0
    count = 0
    precount = 0
    temp = 0

    for i in range(len(testResult)):
        if testResult[i] == testLabels[i]:
            if testResult[i] == 1:
                precount += 1
                temp += 1

            count += 1
        else:
            temp+=1

        
    ourScore = 100*(count/len(testResult))
    precision = 100*(precount/temp)


    print("Accuracy: {0:.2f}%".format(ourScore))
    print("Precision (Ignoring all correctly identified non-fraudulent purchases): {0:.2f}%".format(precision))
    print()
    print("Press Enter to see this decision tree: ")
    input()
    plt.show()