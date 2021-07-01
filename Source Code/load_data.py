import numpy as np
import pandas as pd


def get_tfidf_features(path):
    data_tfidf_raw = pd.read_csv(path, sep=',')
    data_tfidf_feature_raw = data_tfidf_raw['tweet'].tolist()
    data_tfidf_feature_temp = []
    for instance in data_tfidf_feature_raw:
        new_instance_string = "".join(
            list(filter(lambda e: e != '[' and e != '(' and e != ')' and e != ']', list(instance)))).split(',')
        new_instance_number = list(map(float, new_instance_string))
        i = 0
        new_instance_final = []
        while i < len(new_instance_number):
            word_tuple = (int(new_instance_number[i]), new_instance_number[i + 1])
            i = i + 2
            new_instance_final.append(word_tuple)
        data_tfidf_feature_temp.append(new_instance_final)
    data_tfidf_feature = np.zeros((len(data_tfidf_feature_temp), 2038))
    for i in range(len(data_tfidf_feature_temp)):
        for word in data_tfidf_feature_temp[i]:
            data_tfidf_feature[i][word[0]] = word[1]
    return data_tfidf_feature


def get_count_features_ont_hot(path):
    data_count_raw = pd.read_csv(path, sep=',')

    data_count_feature_raw = data_count_raw['tweet'].tolist()
    data_count_feature_temp = []
    for instance in data_count_feature_raw:
        new_instance_string = "".join(
            list(filter(lambda e: e != '[' and e != '(' and e != ')' and e != ']', list(instance)))).split(',')
        new_instance_number = list(map(float, new_instance_string))
        i = 0
        new_instance_final = []
        while i < len(new_instance_number):
            word_tuple = (int(new_instance_number[i]), new_instance_number[i + 1])
            i = i + 2
            new_instance_final.append(word_tuple)
        data_count_feature_temp.append(new_instance_final)

    data_count_feature = np.zeros((len(data_count_feature_temp), 2038))
    for i in range(len(data_count_feature_temp)):
        for word in data_count_feature_temp[i]:
            data_count_feature[i][word[0]] = 1
    return data_count_feature


def get_glove300_features(path):
    data_glove300_raw = pd.read_csv(path, sep=',')

    data_glove300_feature = []
    data_glove300_feature_raw = data_glove300_raw['tweet'].tolist()
    for instance in data_glove300_feature_raw:
        new_instance_string = list(instance.split(' '))
        new_instance_number = list(map(float, new_instance_string))
        data_glove300_feature.append(new_instance_number)
    return data_glove300_feature


def get_labels(path):
    data_raw = pd.read_csv(path, sep=',')
    data_label = data_raw['region'].tolist()
    return data_label


def get_users(path):
    data_raw = pd.read_csv(path, sep=',')
    data_user = data_raw['user'].tolist()
    return data_user
