import numpy as np

from training.training_run import simulate_windows

def add(x,y):
    return x + y

N = 10

widths, heights, x, y = simulate_windows(N = N)



def test_simulate_windows():
    assert np.unique([add(x[i] , widths[i]) <= 4032 for i in range(N)]) == True
    assert np.unique([ add(y[ i ], heights[ i ]) <= 3024 for i in range(N) ]) == True