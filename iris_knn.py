'''
    This script explores distance measures and classification

    Task:
    1) Expand block 1 to classify all samples in the test set.
    2) Calculate the prediction accuracy.
        prediction accuracy = # of accuate predictions / # predictions.
    3) Calculate a confusion matrix for your predictions.
        https://en.wikipedia.org/wiki/Confusion_matrix
        A confusion matrix gives more context to the performance of a classifie.
        The confusion matrix is a table where the columns count predicted labels
        The rows count the actual labels. Correct prediction will line the diagonals.
    4) Try to expland the nearest neighbor classifier to a 3-nearest neighbor classifier.
        Compare your 3-nn to your 1-nn results in terms of both accuracy and confusion matrix.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def distance(a, b):
    return np.sqrt(np.sum((b-a)**2))

def separate_data_labels(dataframe, label_name):
    """
        Separate the data vectors from their label (class/category)
    """
    X = dataframe.drop(columns=label_name)
    Y = dataframe[label_name]
    return X, Y


def main():
    # load training and test data.
    train_df = pd.read_csv('./train_iris.csv')
    test_df = pd.read_csv('./test_iris.csv')

    # separate data matrix from classification lables. See function above.
    train_X, train_Y = separate_data_labels(train_df, 'species')
    test_X, test_Y = separate_data_labels(test_df, 'species')

    # ----------- Block 1 ----------- #
    # begin classification on test set.

    num_accurate_pred = 0 
    for test_index in range(len(train_df)):
        
        test_vector = test_X.iloc[test_index,:]
        true_label = test_Y.iloc[test_index]

        # calculate all distances between test_vector and training data.
        distances = train_X.apply(lambda x:distance(test_vector,x),1)
        # get index of smallest value.
        index = np.argmin(distances)
        # look up the value in the training labels.
        pred_label = train_Y.iloc[index]

        # number of accurate predictions
        if pred_label == true_label:
            num_accurate_pred +=1

        # print index and the labels
        print('training set index:', index, ', predicted:', pred_label, ', actual:', true_label)
            
    print("The prediction accuracy is:",str(round(num_accurate_pred/len(test_df)*100,2) ) )



main()
