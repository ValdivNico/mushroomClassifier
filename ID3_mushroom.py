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


def make_tree(data, classes, features):
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
        bestFeature = np.argmax(gain)
        fi_s = np.unique(np.transpose(data)[bestFeature])
        feat = features.pop(bestFeature)
        tree = {feat: {}}
        # Find the possible feature values
        for fi in fi_s:
            # Find the datapoints with each feature value
            t, d = sub_data(data, classes, bestFeature, fi)
            # Now recurse to the next level
            subtree = make_tree(d, t, features)
            # And on returning, add the subtree on to the tree
            tree[feat][fi] = subtree
        return tree


def main():
    targets = ['t', 'f', 'f', 'f']
    data = [['v', 'x', 'b', 'x'], ['e', 'x', 'b', 'x'], ['x', 'x', 'c', 'x'], ['x', 'x', 'a', 'x']]
    features = [0, 1, 2, 3]
    tree = make_tree(data, targets, features)
    print(tree)


main()
