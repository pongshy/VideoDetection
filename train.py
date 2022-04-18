from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import data
import pickle
import datetime


# train the classifier
def train(classifier_save_path='', datasets_path='./datasets/sport2.csv'):
    datas, target = data.get_datas(datasets_path)
    print(type(datas), datas.shape)
    print(type(target), target.shape)

    X_train, X_test, y_train, y_test = train_test_split(datas, target, test_size=0.3, random_state=104)
    # print(type(y_train), y_train.shape)
    # print(y_train)
    # print(y_test)

    kernal = 'rbf'
    # classifier = svm.SVC(kernel='rbf')
    classifier = svm.SVC(C=8, kernel='rbf', gamma=1, decision_function_shape='ovo')
    # classifier = svm.SVC(kernel='poly')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print('Accuarcy:', metrics.accuracy_score(y_pred, y_test))
    print('Precision:', metrics.precision_score(y_pred, y_test))
    print('Recall:', metrics.recall_score(y_pred, y_test))
    save_model(classifier, classifier_save_path)

    # 打日志
    with open('./log/train.log', 'a') as f:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write('\n' + now + ':\n')
        f.write('Train the sport classfiler completely.\n')
        f.write('kernal: {}.\n'.format(kernal))
        f.write('Store in {}.\n'.format(classifier_save_path))
        f.write('Accuarcy: {}.\n'.format(metrics.accuracy_score(y_pred, y_test)))
        f.write('Precision: {}.\n'.format(metrics.precision_score(y_pred, y_test)))
        f.write('Recall: {}.\n'.format(metrics.recall_score(y_pred, y_test)))
        f.close()
        print('train log has written!')


# 保存训练好的分类器
def save_model(classifier, model_save_path='./model/t.pickle'):
    with open(model_save_path, 'wb') as f:
        f.write(pickle.dumps(classifier))
        print('in train.py:', 'save the trained classifier successfully!')
        f.close()


# 获取已经训练好的分类器
def get_model(model_save_path):
    with open(model_save_path, 'rb') as f:
        classifier = pickle.load(f)
        return classifier


if __name__ == '__main__':
    datasets_path = './datasets/sport4.csv'
    classifier_save_path = './model/sport4.pickle'
    train(classifier_save_path, datasets_path)