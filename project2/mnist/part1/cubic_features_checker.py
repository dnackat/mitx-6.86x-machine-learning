import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
from features import cubic_features

def verify_cubic_features1D():
    X=np.array([[np.sqrt(3)],[0]])
    X_cube=np.sort(cubic_features(X))
    X_correct = np.array([[ 1., np.sqrt(9), np.sqrt(27), np.sqrt(27)],[0., 0., 0., 1.]]);
    
    if np.all(np.absolute(X_cube-X_correct) < 1.0e-6):
        print ("Verifying cubic features of 1 dimension: Passed")
    else:
        print ("Verifying cubic features of 1 dimension: Failed")

    
def verify_cubic_features2D():
    X=np.array([[np.sqrt(3),np.sqrt(3)],[0,0]])
    X_cube=np.sort(cubic_features(X))
    X_correct = np.array([[1., 3., 3., 5.19615242, 5.19615242, 5.19615242, 5.19615242, 7.34846923, 9., 9.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    
    if np.all(np.absolute(X_cube-X_correct) < 1.0e-6):
        print ("Verifying cubic features of 2 dimensions: Passed")
    else:
        print ("Verifying cubic features of 2 dimensions: Failed")
        

def verify_cubic_features2D2():
    X=np.array([[np.sqrt(3),0],[0,np.sqrt(3)]])
    X_cube=np.sort(cubic_features(X))
    X_correct = np.array([[0., 0., 0., 0., 0., 0., 1., 3., 5.19615242, 5.19615242],
                          [0., 0., 0., 0., 0., 0., 1., 3., 5.19615242, 5.19615242]])
    
    if np.all(np.absolute(X_cube-X_correct) < 1.0e-6):
        print ("Verifying cubic features of 2 dimensions asymmetric vectors: Passed")
    else:
        print ("Verifying cubic features of 2 dimensions asymmetric vectors: Failed")

verify_cubic_features1D()
verify_cubic_features2D()
verify_cubic_features2D2()
