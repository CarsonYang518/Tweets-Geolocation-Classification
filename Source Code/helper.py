import numpy as np
from collections import Counter


def majority_votes(model_predictions, data_user):
    user_labels = {}
    user_label = {}
    predictions = []
    for i in range(len(data_user)):
        if data_user[i] not in user_labels:
            user_labels[data_user[i]] = list(model_predictions[i])
        else:
            user_labels[data_user[i]].append(model_predictions[i])

    for key in user_labels.keys():
        countMap = Counter(user_labels[key])
        sortedCountMap = sorted(countMap.items(), key=lambda item: item[1])
        user_label[key] = sortedCountMap[-1][0]

    for i in range(len(data_user)):
        predictions.append(user_label[data_user[i]])

    return predictions


def max_average_probability(model_probs_predictions, data_user, classes):
    user_predict_probs = {}
    user_predict_probs_average = {}
    predictions = []
    for i in range(len(data_user)):
        if data_user[i] not in user_predict_probs:
            user_predict_probs[data_user[i]] = list()
            user_predict_probs[data_user[i]].append(list(model_probs_predictions[i]))
        else:
            user_predict_probs[data_user[i]].append(list(model_probs_predictions[i]))

    for key in user_predict_probs.keys():
        user_predict_probs_average[key] = [0.0, 0.0, 0.0, 0.0]
        for prob in user_predict_probs[key]:
            user_predict_probs_average[key][0] += prob[0]
            user_predict_probs_average[key][1] += prob[1]
            user_predict_probs_average[key][2] += prob[2]
            user_predict_probs_average[key][3] += prob[3]
        user_predict_probs_average[key][0] /= len(user_predict_probs[key])
        user_predict_probs_average[key][1] /= len(user_predict_probs[key])
        user_predict_probs_average[key][2] /= len(user_predict_probs[key])
        user_predict_probs_average[key][3] /= len(user_predict_probs[key])

    for i in range(len(data_user)):
        temp = np.array(user_predict_probs_average[data_user[i]])
        predictions.append(classes[temp.argmax()])

    return predictions