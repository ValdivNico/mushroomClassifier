import numpy as np
import csv
import sys


def calc_entropy(p):
    if p != 0:
        return -p * np.log2(p)
    else:
        return 0


def calc_feature_gain(data, classes, feature):
    gain = 0
    nData = len(data)

    # compute all different values in column feature
    fi_s = {}
    for row in data:
        if row[feature] not in fi_s.keys():
            fi_s[row[feature]] = 1
        else:
            fi_s[row[feature]] += 1

    for fi in fi_s.keys():
        fi_entropy = 0
        row_indx = 0
        newClasses = {}
        classCounts = 0
        for row in data:
            if row[feature] == fi:
                classCounts += 1
                if classes[row_indx] in newClasses.keys():
                    newClasses[classes[row_indx]] += 1
                else:
                    newClasses[classes[row_indx]] = 1
            row_indx += 1

        for aclass in newClasses.keys():
            fi_entropy += calc_entropy(float(newClasses[aclass]) / classCounts)

        gain += float(fi_s[fi]) / nData * fi_entropy
    return gain


def calc_total_entropy(targets):
    diff_targets = {}
    n = len(targets)
    for t in targets:
        if t not in diff_targets:
            diff_targets[t] = 1
        else:
            diff_targets[t] += 1
    entropy = 0
    for t in diff_targets:
        p = diff_targets[t] / float(n)
        entropy += calc_entropy(p)

    return entropy


def sub_data(data, targets, feature, fi):
    new_data = []
    new_targets = []
    nFeatures = len(data[0])
    row_idx = 0
    for row in data:
        if row[feature] == fi:
            if feature == 0:
                new_row = row[1:]
            elif feature == nFeatures:
                new_row = row[:-1]
            else:
                new_row = row[:feature]
                new_row.extend(row[feature + 1:])

            new_data.append(new_row)
            new_targets.append(targets[row_idx])
        row_idx += 1

    return new_targets, new_data


def make_tree(data, classes, features):
    """
    Generate a decision tree using the ID3 algorithm.
    :param data: 
    :param classes: 
    :param features: 
    :return: 
    """
    nData = len(data)
    nFeatures = len(features)
    # Have reached an empty branch
    uniqueT = {}
    for aclass in classes:
        if aclass in uniqueT.keys():
            uniqueT[aclass] += 1
        else:
            uniqueT[aclass] = 1

    default = max(uniqueT, key=uniqueT.get)
    if nData == 0 or nFeatures == 0:
        return default
    elif len(np.unique(classes)) == 1:
        # Only 1 class remains
        return classes[0]
    else:
        # Choose which feature is best
        totalEntropy = calc_total_entropy(classes)
        gain = np.zeros(nFeatures)
        for feature in range(nFeatures):
            g = calc_feature_gain(data, classes, feature)
            gain[feature] = totalEntropy - g
        best = np.argmax(gain)  # index of the best feature
        fi_s = np.unique(np.transpose(data)[best])
        feature = features.pop(best)    # Feature Name at that "best" position
        tree = {feature: {}}
        # Find the possible feature values
        for fi in fi_s:
            # Find the datapoints with each feature value
            t, d = sub_data(data, classes, best, fi)
            # Now recurse to the next level
            subtree = make_tree(d, t, features)
            # And on returning, add the subtree on to the tree
            tree[feature][fi] = subtree
        return tree


def print_tree(tree):
    """
    :param tree: a dictionary tree implementation
    :return: prints a better visualization of the tree
    """
    root = tree.keys()
    for i in tree.keys():
        print(i)
        for k in tree[i]:
            print(k)


def import_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        raw_data = list(reader)
    features = raw_data.pop(0)
    features.remove("class")
    targets = [row.pop(0) for row in raw_data]

    test1 = len(raw_data[0]) == len(features)
    test2 = len(raw_data) == len(targets)
    if test1 and test2:
        return raw_data, targets, features
    else:
        print("ERROR in data processing")
        exit(-1)


def main():
    csv_file = sys.argv[1]
    data, targets, features = import_data(filename=csv_file)
    tree = make_tree(data, targets, features)
    print(tree)
    #print_tree(tree)


main()
