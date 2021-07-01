from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd


def zero_r_baseline(train_data_y, dev_data_x, dev_data_y, test_data_x):
    zero_r_predictions = []
    countMap = Counter(train_data_y)
    sortedCountMap = sorted(countMap.items(), key=lambda item: item[1])
    majority_class = sortedCountMap[-1][0]
    for i in range(len(dev_data_y)):
        zero_r_predictions.append(majority_class)
    acc = accuracy_score(dev_data_y, zero_r_predictions)
    f1 = f1_score(dev_data_y, zero_r_predictions, average='macro')
    test_zero_r_predictions = []
    for i in range(len(test_data_x)):
        test_zero_r_predictions.append(majority_class)
    # file = pd.DataFrame(data=test_zero_r_predictions)
    # file.to_csv('0r_test.csv',encoding='gbk')
    return acc, f1

