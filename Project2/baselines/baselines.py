from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from evaluation import single_label_accuracy, generate_metric
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

classifiers = {
    "SVM": svm.SVC(C=7.5, kernel="rbf", gamma="auto"),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "RandomForest": RandomForestClassifier(n_estimators=500, criterion="gini"),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1)
}

def sml_train(config):
    train_set = config["raw_train_set"]
    valid_set = config["raw_valid_set"]
    tx, ty = train_set[:, :-2], train_set[:, -1]
    vx, vy = valid_set[:, :-2], valid_set[:, -1]
    for key, clf in classifiers.items():
        print(key)
        clf.fit(tx, ty)
        print("Train Acc:", clf.score(tx, ty), "Valid Acc:", clf.score(vx, vy))
        pvy = clf.predict(vx)
        acc_result = None
        acc_result = single_label_accuracy(pvy, vy, acc_result)
        print(generate_metric(acc_result))