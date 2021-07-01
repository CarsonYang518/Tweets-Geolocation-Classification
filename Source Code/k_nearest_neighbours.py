from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import load_data
import pandas as pd
import joblib


def knn(train_data_x, train_data_y, dev_data_x, dev_data_y, test_data_x):
    # accs = []
    # for i in range(5, 500):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(train_data_x, train_data_y)
    #     predictions = knn.predict(dev_data_x)
    #     acc = accuracy_score(dev_data_y, predictions)
    #     accs.append(acc)
    #     print(acc)
    # file = pd.DataFrame(data=accs)
    # file.to_csv('knn_accs.csv', encoding='gbk')

    knn = KNeighborsClassifier(n_neighbors=330)
    knn.fit(train_data_x, train_data_y)
    predictions = knn.predict(dev_data_x)
    acc = accuracy_score(dev_data_y, predictions)
    f1 = f1_score(dev_data_y, predictions, average='macro')
    test_predictions = knn.predict(test_data_x)
    # file = pd.DataFrame(data=test_predictions)
    # file.to_csv('knn_test.csv',encoding='gbk')
    return acc, f1