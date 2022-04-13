#import scikit-learn dataset library
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pickle


def train():
    cancer = datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)
    # print("X_train:", len(X_train))
    # print("X_test:", len(X_test))
    # print("y_train:", len(y_train))
    # print("y_test", len(y_test))
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # 读取模型
    # f = open(r"output/test.pickle", 'rb')
    # clf = pickle.load(f)

    y_pred = clf.predict(X_test)

    # 保存模型
    # f = open(r'output/test.pickle', 'wb')
    # f.write(pickle.dumps(clf))
    # f.close()

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))


if __name__ == '__main__':
    # cancer = datasets.load_breast_cancer()
    #
    # print("Features:", cancer.feature_names, len(cancer.feature_names))
    #
    # print("Labels:", cancer.target_names)
    #
    # print(cancer.data.shape)
    #
    # print(cancer.data[0:5], len(cancer.data[0]))
    #
    # print(cancer.target)

    train()



