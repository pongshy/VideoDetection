#import scikit-learn dataset library
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pickle
import csv
import pandas as pd
import numpy as np


def train():
    cancer = datasets.load_breast_cancer()
    print(type(cancer.data), cancer.data.shape)

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)
    print(type(X_train), X_train.shape)
    print(type(X_test), X_test.shape)
    print(type(y_test), y_train.shape)
    print("X_train:", len(X_train))
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


# 写入csv
def write_in_csv(file):
    # 打开一个csv文件用于写入
    file = open(file, 'w+', encoding='utf8', newline='')

    # 获取csv的writer用于写入数据
    writer = csv.writer(file)

    # 写入第一行作为列名
    writer.writerow(('id', 'name', 'age'))
    # 写入一行数据
    writer.writerow(('0', 'none', '1'))
    # 写入多行数据
    writer.writerows([['1', '我们', '2'], ['2', 'ab', '100'], ['3', 'abc', '4']])


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

    # train()
    # d = dict()
    # print(type(d.keys()))

    data_file = './datasets/t.csv'
    # 读取csv
    data = pd.read_csv(data_file, header=None)
    print(data)
    print(type(data))
    data = np.array(data)
    print(data)
    print(type(data))



