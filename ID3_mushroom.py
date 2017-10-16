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
def calc_feature_gain(data, classes, feature):
    gain = 0
    nData = len(data)

    # compute all different values in column feature
    fi_s = []
    for row in data:
        if row[feature] not in fi_s:
            fi_s.append(row[feature])

    # initialize structures:
    count_fi_s = np.zeros(len(fi_s))  # hold the count of different fi_s
    entropy = np.zeros(len(fi_s))  # entropy for each fi_s
    fi_indx = 0  # position holder

    for fi in fi_s:
        row_indx = 0
        newClasses = []
        for row in data:
            if row[feature] == fi:
                count_fi_s[fi_indx] += 1

                newClasses.append(classes[row_indx])
            row_indx += 1

        classVal = []
        for aclass in newClasses:
            if classVal.count(aclass) == 0:
                classVal.append(aclass)

        classCounts = np.zeros(len(classVal))
        class_indx = 0
        for classvalue in classVal:
            for aclass in newClasses:
                if aclass == classvalue:
                    classCounts[class_indx] += 1
            class_indx += 1

        for class_indx in range(len(classVal)):
            entropy[fi_indx] += calc_entropy(float(classCounts[class_indx]) / sum(classCounts))
        gain += float(count_fi_s[fi_indx]) / nData * entropy[fi_indx]
        fi_indx += 1
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


def make_tree(data, classes, featureNames, totalEntropy):
    nData = len(data)
    nFeatures = len(featureNames)
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
            for datapoint in data:
                if datapoint[bestFeature] == value:
                    if bestFeature == 0:
                        datapoint = datapoint[1:]
                        newNames = featureNames[1:]
                    elif bestFeature == nFeatures:
                        datapoint = datapoint[:-1]
                        newNames = featureNames[:-1]
                    else:
                        datapoint = datapoint[:bestFeature]
                        datapoint.extend(datapoint[bestFeature + 1:])
                        newNames = featureNames[:bestFeature]
                        newNames.extend(featureNames[bestFeature + 1:])
                    newData.append(datapoint)
                    newClasses.append(classes[index])
                index += 1
            # Now recurse to the next level
            subtree = make_tree(newData, newClasses, newNames, totalEntropy)
            # And on returning, add the subtree on to the tree
            tree[featureNames[bestFeature]][value] = subtree
        return tree


def main():
    targets = ['t', 'f', 'f', 'f']
    total_entropy = calc_total_entropy(targets)
    data = [[1, 1, 'b', 1], [1, 1, 'b', 1], [1, 1, 'c', 1], [1, 1, 'a', 1]]
    feature = [1, 2, 3, 4]

    tree = make_tree(data, targets, feature, total_entropy)

    print(total_entropy - g)


main()
