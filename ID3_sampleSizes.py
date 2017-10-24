import numpy as np
import csv
import copy


def calc_entropy(p):
    if p != 0:
        return -p * np.log2(p)
    else:
        return 0


def calc_feature_gain(dataset, classes, feature):
    gain = 0
    nData = len(dataset)

    # compute all different values in column feature
    fi_s = {}
    for row in dataset:
        if row[feature] not in fi_s.keys():
            fi_s[row[feature]] = 1
        else:
            fi_s[row[feature]] += 1

    for fi in fi_s.keys():
        fi_entropy = 0
        row_indx = 0
        newClasses = {}
        classCounts = 0
        for row in dataset:
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


def sub_data(dataset, targets, feature, fi):
    """
    Calculates the subset of data that contains
    :param dataset: The training dataset
    :param targets: list of targets
    :param feature: feature index
    :param fi: fixed value of the feature
    :return: subset of data that contains a particular value for a fixed feature
    """
    new_data = []
    new_targets = []
    nFeatures = len(dataset[0])
    row_idx = 0
    for row in dataset:
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


def make_tree(dataset, classes, features):
    """
    Generate a decision tree using the ID3 algorithm.
    :param dataset: 
    :param classes: 
    :param features: 
    :return: 
    """
    nData = len(dataset)
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
            g = calc_feature_gain(dataset, classes, feature)
            gain[feature] = totalEntropy - g
        best = np.argmax(gain)  # index of the best feature
        fi_s = np.unique(np.transpose(dataset)[best])
        feature = features.pop(best)  # Feature Name at that "best" position
        tree = {feature: {}}
        # Find the possible feature values
        for fi in fi_s:
            # Find the datapoints with each feature value
            t, d = sub_data(dataset, classes, best, fi)
            # Now recurse to the next level
            subtree = make_tree(d, t, features)
            # And on returning, add the subtree on to the tree
            tree[feature][fi] = subtree
        return tree


def print_tree(tree, answer):
    """
    :param tree: a dictionary tree implementation
    :return: prints a better visualization of the tree
    origin3 {feature: leaves]
    origin2 {feature: origin3, leaves]
    origin1 {feature: origin2, leaves]
    root {feature: origin1, leaves]

    """
    tree_imp = {}
    for root in tree.keys():
        level = []
        leaves = {}
        next_keys = []
        for k in tree[root]:
            if type(tree[root][k]) is not dict:
                leaf = tree[root][k]
                if leaf in leaves:
                    leaves[leaf].append(k)
                else:
                    leaves[leaf] = [k]
            else:
                next_keys.append(k)
                print_tree(tree[root][k], k)

        if len(next_keys) != 0:
            level.extend(next_keys)
        level.append(leaves)
        tree_imp[root] = level
        print(answer, tree_imp)


def training(dataset, features):
    """
    :param dataset: The training dataset including targets
    :param features: list of features
    :return: a tree produced by ID3 algorithm
    """
    targets = [row.pop(0) for row in dataset]
    tree = make_tree(dataset, targets, features)
    return tree


def tree_output(tree, data_row, features):
    """
    :param tree: the decision tree resulting from training
    :param data_row: an observation in the testing dataset
    :param features: list of features
    :return: target of the branch followed by the row or if the feature key was never observed, then it returns "N/A"
    """
    if type(tree) is not dict:
        return str(tree)
    else:
        for key in tree.keys():
            f_idx = features.index(key)
            f_i = data_row[f_idx]
            try:
                return tree_output(tree[key][f_i], data_row, features)
            except KeyError as ke:
                return "N/A"


def validator_seq(dataset, features, tree):
    """
    :param dataset: The testing dataset including targets
    :param features: a list of feature names
    :param tree: a decision tree
    :return: percentage of correct classifications, and the number of unknown classifications.
    """
    targets = [row.pop(0) for row in dataset]
    good = 0
    unknowns = 0
    n = len(dataset)
    row_indx = 0

    for row in dataset:
        output = tree_output(tree, row, features)
        if output == targets[row_indx]:
            good += 1
        elif output == "N/A":
            unknowns += 1
        row_indx += 1
    return good / float(n), unknowns


def import_data(filename, targetFeat):
    """
    Importing the data from csv
    :param filename: the name of the csv file
    :return: the dataset including targets as a list of lists and a list of feature names
    """
    with open(file=filename, mode='r') as f:
        reader = csv.reader(f)
        dataset = list(reader)
    features = dataset.pop(0)
    features.remove(targetFeat)
    if len(dataset[0])-1 == len(features):
        return dataset, features
    else:
        print("ERROR in data processing")
        exit(-1)


def run_tests(train_alphas, dataset, features):
    nData = len(dataset)
    Scores = []
    for i in range(10):
        data_perm = (np.random.permutation(dataset)).tolist()
        for train_alpha in train_alphas:
            feat_train = copy.deepcopy(features)
            feat_test = copy.deepcopy(features)
            train_size = int(np.ceil(nData * train_alpha))
            train_data = copy.deepcopy(data_perm[0:train_size])
            test_data = copy.deepcopy(data_perm[train_size:nData])
            decision_tree = training(dataset=train_data, features=feat_train)
            perc_correct, num_unknown = validator_seq(dataset=test_data, features=feat_test, tree=decision_tree)
            row = [train_alpha, i, perc_correct, num_unknown]
            Scores.append(row)
    return Scores


def main():
    """
    csv_file contains the name of the data in csv format.
    alphas is a list of percentages to run test on each percent amount of training data
    :return: prints the Scores -- can be changed to print or export as csv
    """
    csv_file = "mushrooms.csv"
    data, featName = import_data(filename=csv_file, targetFeat="class")
    alphas = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    scores = run_tests(alphas, data, featName)
    print(scores)


main()
