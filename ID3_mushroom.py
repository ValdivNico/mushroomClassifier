import numpy as np


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

'''
def make_tree(data, classes, featureNames, totalEntropy):
    nData = len(data)
    nFeatures = len(featureNames)
    frequency = np.unique(classes, return_counts=True)
    default = classes[np.argmax(frequency)]
    if nData == 0 or nFeatures == 0:
        # Have reached an empty branch
        return default
    elif classes.count(classes[0]) == nData:
        # Only 1 class remains
        return classes[0]
    else:
        # Choose which feature is best
        gain = np.zeros(nFeatures)
        for feature in range(1, nFeatures):
            g = calc_feature_gain(data, classes, feature)
            gain[feature] = totalEntropy - g
        bestFeature = np.argmax(gain)
        tree = {featureNames[bestFeature]: {}}
        # Find the possible feature values
        for value in values:
            # Find the datapoints with each feature value

            # Now recurse to the next level
            subtree = make_tree(newData, newClasses, newNames, totalEntropy)
            # And on returning, add the subtree on to the tree
            tree[featureNames[bestFeature]][value] = subtree
        return tree
'''
def main():
    targets = ['t', 'f', 'f', 'f']
    total_entropy = calc_total_entropy(targets)
    data = [[1, 1, 'b', 1], [1, 1, 'b', 1], [1, 1, 'c', 1], [1, 1, 'a', 1]]
    feature = [1, 2, 3, 4]
    f_gain = calc_feature_gain(data, targets, feature[1])
    #tree = make_tree(data, targets, feature, total_entropy)
    print(np.unique(targets))
    print(sub_data(data, targets, 2, 'b'))
    print (total_entropy - f_gain)


main()
