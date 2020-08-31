import numpy as np

def euc_distance(X,Y):

    distance = np.sum([(X[i] - Y[i])**2 for i in range(len(X))])

    return distance





if "__name__" == "__main__":
    X = np.array([ 1, 2, 3 ])

    print(euc_distance(X, X) == 0)
