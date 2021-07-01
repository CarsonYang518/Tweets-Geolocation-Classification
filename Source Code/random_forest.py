from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import load_data
import pandas as pd
import joblib
import helper

# accs = []
# for i in range(5, 51):
#     model = RandomForestClassifier(max_depth=i,max_features='sqrt',n_estimators=100, n_jobs=-1)
#     model.fit(train_data_x_tfidf,train_data_y_tfidf)
#     predictions = model.predict(dev_data_x_tfidf)
#     acc = accuracy_score(dev_data_y_tfidf,predictions)
#     accs.append(acc)
#     print(acc)
# file = pd.DataFrame(data=accs)
# file.to_csv('rf_accs.csv', encoding='gbk')

# model = RandomForestClassifier(max_depth=40,max_features='sqrt',n_estimators=100, n_jobs=-1, verbose=2)
# model.fit(train_data_x_tfidf,train_data_y_tfidf)
# joblib.dump(model, 'rf.model')


def random_forest(train_data_x, train_data_y, dev_data_x, dev_data_y, test_data_x):
    model = joblib.load('rf.model')
    predictions = model.predict(dev_data_x)
    acc = accuracy_score(dev_data_y, predictions)
    f1 = f1_score(dev_data_y, predictions, average='macro')
    test_predictions = model.predict(test_data_x)
    # file = pd.DataFrame(data=test_predictions)
    # file.to_csv('rf_test.csv',encoding='gbk')
    return acc, f1


def random_forest_with_majority_votes(train_data_x, train_data_y,
                                     dev_data_x, dev_data_y, dev_data_user, test_data_x, test_data_user):
    model = joblib.load('rf.model')
    model_predictions = model.predict(dev_data_x)
    predictions = helper.majority_votes(model_predictions, dev_data_user)
    acc = accuracy_score(dev_data_y, predictions)
    f1 = f1_score(dev_data_y, predictions, average='macro')
    test_model_predictions = model.predict(test_data_x)
    test_predictions = helper.majority_votes(test_model_predictions, test_data_user)
    # file = pd.DataFrame(data=test_predictions)
    # file.to_csv('rf_majority_votes_test.csv', encoding='gbk')
    return acc, f1


def random_forest_max_average_prob(train_data_x, train_data_y,
                                       dev_data_x, dev_data_y, dev_data_user, test_data_x, test_data_user):
    model = joblib.load('rf.model')
    classes = model.classes_.tolist()
    model_prob_predictions = model.predict_proba(dev_data_x)
    predictions = helper.max_average_probability(model_prob_predictions, dev_data_user, classes)
    acc = accuracy_score(dev_data_y, predictions)
    f1 = f1_score(dev_data_y, predictions, average='macro')
    test_model_prob_predictions = model.predict_proba(test_data_x)
    test_predictions = helper.max_average_probability(test_model_prob_predictions, test_data_user, classes)
    # file = pd.DataFrame(data=test_predictions)
    # file.to_csv('rf_max_average_prob_test.csv', encoding='gbk')
    return acc, f1
