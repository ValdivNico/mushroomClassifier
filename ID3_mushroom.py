import numpy as np


def calc_entropy(p):
    if p != 0:
        return -p * np.log2(p)
    else:
        return 0

# S is sample --> the sample set from the dataset
# F is feature (column)
# f1, f2, ..., fk --> different values in the column F
# S_fi is the set of observations (rows) of the sample which values of F are fi.
# class --> is the target. For us: e or p
# classes might be the target vector
# class values --> the forloop can be replaced by [e, p]
# class count --> we can split newClasses into e's and p's lists and assign the length
def calc_info_gain(data, )
