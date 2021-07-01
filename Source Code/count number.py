import load_data
from collections import Counter

train_data_x_tfidf = load_data.get_tfidf_features('data/train_tfidf.csv')
train_data_y_tfidf = load_data.get_labels('data/train_tfidf.csv')
dev_data_x_tfidf = load_data.get_tfidf_features('data/dev_tfidf.csv')
dev_data_y_tfidf = load_data.get_labels('data/dev_tfidf.csv')
test_data_x_tfidf = load_data.get_tfidf_features('data/test_tfidf.csv')

print(len(train_data_x_tfidf))
print(len(dev_data_x_tfidf))
print(len(test_data_x_tfidf))
print(Counter(train_data_y_tfidf))
print(Counter(dev_data_y_tfidf))

