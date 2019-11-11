import numpy as np


# When you turn this function in to Gradescope, it is easiest to copy and paste this cell to a new python file called hw1.py
# and upload that file instead of the full Jupyter Notebook code (which will cause problems for Gradescope)
def compute_features(names):
    """
    Given a list of names of length N, return a numpy matrix of shape (N, 260)
    with the features described in problem 2b of the homework assignment.
    
    Parameters
    ----------
    names: A list of strings
        The names to featurize, e.g. ["albert einstein", "marie curie"]
    
    Returns
    -------
    numpy.array:
        A numpy array of shape (N, 260)
    """
    Alphabet = ['a', 'b', 'c', 'd', 'e','f', 'g', 'h', 'i', 'j','k', 'l', 'm', 'n', 'o',
               'p', 'q', 'r', 's', 't','u', 'v', 'w', 'x', 'y' , 'z']
 
    N = len(names)
    Feature_matrix = np.zeros((N, 260))
    for row in range(0, N):
        firstLast = names[row].split()
        first = firstLast[0]  #First Name
        last = firstLast[1]   #Last Name
        if(len(first) < 5):
            firstRange = len(first)
        else:
            firstRange = 5
        if(len(last) < 5):
            lastRange = len(last)
        else:
            lastRange = 5
        for index in range(0,firstRange):  #iterate though first 5 letters of First name
            offset = 26 * index
            featureIndex = offset + Alphabet.index(first[index])
            Feature_matrix[row,featureIndex] = 1
        index = 4          
        for Lastindex in range(0,lastRange):  #iterate though first 5 letters of Last name
            index += 1
            offset = 26 * index
            featureIndex = offset + Alphabet.index(last[Lastindex])
            Feature_matrix[row,featureIndex] = 1
    return Feature_matrix