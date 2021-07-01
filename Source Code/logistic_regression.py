from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import load_data
import joblib
import pandas as pd
import helper
import numpy as np


def logistic_regression(train_data_x, train_data_y, dev_data_x, dev_data_y, test_data_x):
    # model = LogisticRegression(multi_class='ovr', verbose=True, solver='saga', max_iter=10000)
    # model.fit(train_data_x_tfidf, train_data_y_tfidf)
    # joblib.dump(model, 'lr_ovr_saga.model')
    model = joblib.load('lr_ovr_saga.model')
    predictions = model.predict(dev_data_x)
    acc = accuracy_score(dev_data_y, predictions)
    f1 = f1_score(dev_data_y, predictions, average='macro')
    test_predictions = model.predict(test_data_x)
    # file = pd.DataFrame(data=test_predictions)
    # file.to_csv('lr_test.csv',encoding='gbk')
    return acc, f1


def logistic_regression_with_majority_votes(train_data_x, train_data_y,
                                     dev_data_x, dev_data_y, dev_data_user, test_data_x, test_data_user):
    # model = LogisticRegression(multi_class='ovr', verbose=True, solver='saga', max_iter=10000)
    # model.fit(train_data_x_tfidf, train_data_y_tfidf)
    # joblib.dump(model, 'lr_ovr_saga.model')
    model = joblib.load('lr_ovr_saga.model')
    model_predictions = model.predict(dev_data_x)
    predictions = helper.majority_votes(model_predictions, dev_data_user)
    acc = accuracy_score(dev_data_y, predictions)
    f1 = f1_score(dev_data_y, predictions, average='macro')
    test_model_predictions = model.predict(test_data_x)
    test_predictions = helper.majority_votes(test_model_predictions, test_data_user)
    # file = pd.DataFrame(data=test_predictions)
    # file.to_csv('lr_majority_votes_test.csv', encoding='gbk')
    return acc, f1


def logistic_regression_max_average_prob(train_data_x, train_data_y,
                                       dev_data_x, dev_data_y, dev_data_user, test_data_x, test_data_user):
    # model = LogisticRegression(multi_class='ovr', verbose=True, solver='saga', max_iter=10000)
    # model.fit(train_data_x_tfidf, train_data_y_tfidf)
    # joblib.dump(model, 'lr_ovr_saga.model')
    model = joblib.load('lr_ovr_saga.model')
    classes = model.classes_.tolist()
    model_prob_predictions = model.predict_proba(dev_data_x)
    predictions = helper.max_average_probability(model_prob_predictions, dev_data_user, classes)
    acc = accuracy_score(dev_data_y, predictions)
    f1 = f1_score(dev_data_y, predictions, average='macro')
    test_model_prob_predictions = model.predict_proba(test_data_x)
    test_predictions = helper.max_average_probability(test_model_prob_predictions, test_data_user, classes)
    # file = pd.DataFrame(data=test_predictions)
    # file.to_csv('lr_max_average_prob_test.csv', encoding='gbk')
    return acc, f1




