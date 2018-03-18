from sklearn import tree
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, brier_score_loss
import csv
import graphviz
import progressbar
import sys

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def train(filename, target):
    print "## Building new model from:", filename

    features_names, features, labels = load_csv(filename)

    print "## Fitting the data"

    clf = tree.DecisionTreeClassifier()
    clf.fit(features, labels)

    if target[-4:] != ".pkl":
        target += ".pkl"

    print "## Saving to:", target
    joblib.dump(clf, target)

def test_data(model, test_data):
    print "## Loading model:", model
    clf = load_model(model)

    features_names, features, labels = load_csv(test_data)

    print "## Running test"

    predictions = clf.predict(features)

    print("## Precision: %1.3f" % precision_score(labels, predictions))
    print("## Recall: %1.3f" % recall_score(labels, predictions))
    print("## F1: %1.3f\n" % f1_score(labels, predictions))

def load_csv(filename):
    num_records = sum(1 for line in open(filename)) - 1
    print "## Collecting", num_records, "rows"

    f = open(filename, 'rb')
    reader = csv.reader(f)
    feature_names = reader.next()

    i = 0
    features = []
    labels = []
    with progressbar.ProgressBar(max_value=num_records, redirect_stdout=True) as bar:
        for row in reader:
            features.append(map(num, row[:-1]))
            labels.append(num(row[-1]))
            i += 1
            bar.update(i)

    f.close()

    return [feature_names, features, labels]

def load_feature_names_csv(filename):
    f = open(filename, 'rb')
    reader = csv.reader(f)
    feature_names = reader.next()
    f.close()

    return feature_names[:-1]

def draw_tree(csv, model, target_names, target):
    clf = load_model(model)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=load_feature_names_csv(csv),
                                    class_names=target_names,
                                    filled=True, rounded=True,
                                    special_characters=True,
                                    max_depth=5)
    graph = graphviz.Source(dot_data)
    graph.render(target)

def load_model(filename):
    if filename[-4:] != ".pkl":
        filename += ".pkl"

    return joblib.load(filename)

def main():
    arg1 = sys.argv[1]

    if arg1 == "train":
        csv = sys.argv[2]
        target = sys.argv[3]
        train(csv, target)
    elif arg1 == "test":
        model = sys.argv[2]
        test_csv = sys.argv[3]
        test_data(model, test_csv)
    elif arg1 == "tree":
        csv = sys.argv[2]
        model = sys.argv[3]
        target_names = sys.argv[4].split(',')
        target = sys.argv[5]
        draw_tree(csv, model, target_names, target)
    else:
        print "## Arg not recognized, try:\n", "## bm"

if __name__ == "__main__":
    main()
