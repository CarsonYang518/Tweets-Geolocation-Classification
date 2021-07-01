from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import helper


def native_bayes(train_data_x, train_data_y, dev_data_x, dev_data_y, test_data_x):
    model = BernoulliNB()
    model.fit(train_data_x, train_data_y)
    predictions = model.predict(dev_data_x)
    acc = accuracy_score(dev_data_y, predictions)
    f1 = f1_score(dev_data_y, predictions, average='macro')
    test_predictions = model.predict(test_data_x)
    # file = pd.DataFrame(data=test_predictions)
    # file.to_csv('nb_test.csv',encoding='gbk')
    return acc, f1


def native_bayes_with_majority_votes(train_data_x, train_data_y,
                                     dev_data_x, dev_data_y, dev_data_user, test_data_x, test_data_user):
    model = BernoulliNB()
    model.fit(train_data_x, train_data_y)
    model_predictions = model.predict(dev_data_x)
    predictions = helper.majority_votes(model_predictions, dev_data_user)
    acc = accuracy_score(dev_data_y, predictions)
    f1 = f1_score(dev_data_y, predictions, average='macro')
    test_model_predictions = model.predict(test_data_x)
    test_predictions = helper.majority_votes(test_model_predictions, test_data_user)
    # file = pd.DataFrame(data=test_predictions)
    # file.to_csv('nb_majority_votes_test.csv', encoding='gbk')
    return acc, f1


def native_bayes_with_max_average_prob(train_data_x, train_data_y,
                                       dev_data_x, dev_data_y, dev_data_user, test_data_x, test_data_user):
    model = BernoulliNB()
    model.fit(train_data_x, train_data_y)
    model_prob_predictions = model.predict_proba(dev_data_x)
    classes = model.classes_.tolist()
    predictions = helper.max_average_probability(model_prob_predictions, dev_data_user, classes)
    acc = accuracy_score(dev_data_y, predictions)
    f1 = f1_score(dev_data_y, predictions, average='macro')
    test_model_prob_predictions = model.predict_proba(test_data_x)
    test_predictions = helper.max_average_probability(test_model_prob_predictions, test_data_user, classes)
    # file = pd.DataFrame(data=test_predictions)
    # file.to_csv('nb_max_average_prob_test.csv', encoding='gbk')
    return acc, f1
