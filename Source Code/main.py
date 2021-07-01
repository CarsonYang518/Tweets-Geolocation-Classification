import load_data
import native_bayes
import k_nearest_neighbours
import zero_r_baseline
import logistic_regression
import random_forest

train_data_x_one_hot = load_data.get_count_features_ont_hot('data/train_count.csv')
train_data_y_one_hot = load_data.get_labels('data/train_count.csv')
dev_data_x_one_hot = load_data.get_count_features_ont_hot('data/dev_count.csv')
dev_data_y_one_hot = load_data.get_labels('data/dev_count.csv')
dev_data_user_one_hot = load_data.get_users('data/dev_count.csv')
test_data_x_one_hot = load_data.get_count_features_ont_hot('data/test_count.csv')
test_data_user_one_hot = load_data.get_users('data/test_count.csv')

train_data_x_tfidf = load_data.get_tfidf_features('data/train_tfidf.csv')
train_data_y_tfidf = load_data.get_labels('data/train_tfidf.csv')
dev_data_x_tfidf = load_data.get_tfidf_features('data/dev_tfidf.csv')
dev_data_y_tfidf = load_data.get_labels('data/dev_tfidf.csv')
dev_data_user_tfidf = load_data.get_users('data/dev_tfidf.csv')
test_data_x_tfidf = load_data.get_tfidf_features('data/test_tfidf.csv')
test_data_user_tfidf = load_data.get_users('data/test_tfidf.csv')

acc1, f11 = native_bayes.native_bayes(train_data_x_one_hot, train_data_y_one_hot, dev_data_x_one_hot, dev_data_y_one_hot, test_data_x_one_hot)
acc2, f12 = native_bayes.native_bayes_with_majority_votes(train_data_x_one_hot, train_data_y_one_hot, dev_data_x_one_hot, dev_data_y_one_hot, dev_data_user_one_hot,
                                                          test_data_x_one_hot, test_data_user_one_hot)
acc3, f13 = native_bayes.native_bayes_with_max_average_prob(train_data_x_one_hot, train_data_y_one_hot, dev_data_x_one_hot, dev_data_y_one_hot, dev_data_user_one_hot,
                                                            test_data_x_one_hot, test_data_user_one_hot)
acc4, f14 = k_nearest_neighbours.knn(train_data_x_one_hot, train_data_y_one_hot, dev_data_x_one_hot, dev_data_y_one_hot, test_data_x_one_hot)
acc5, f15 = zero_r_baseline.zero_r_baseline(train_data_y_one_hot, dev_data_x_one_hot, dev_data_y_one_hot, test_data_x_one_hot)

acc6,f16 = logistic_regression.logistic_regression(train_data_x_tfidf, train_data_y_tfidf, dev_data_x_tfidf, dev_data_y_tfidf, test_data_x_tfidf)
acc7,f17 = logistic_regression.logistic_regression_with_majority_votes(train_data_x_tfidf, train_data_y_tfidf, dev_data_x_tfidf, dev_data_y_tfidf, dev_data_user_tfidf,
                                                                       test_data_x_tfidf,test_data_user_tfidf)
acc8,f18 = logistic_regression.logistic_regression_max_average_prob(train_data_x_tfidf, train_data_y_tfidf, dev_data_x_tfidf, dev_data_y_tfidf, dev_data_user_tfidf,
                                                                       test_data_x_tfidf,test_data_user_tfidf)

acc9,f19 = random_forest.random_forest(train_data_x_tfidf, train_data_y_tfidf, dev_data_x_tfidf, dev_data_y_tfidf, test_data_x_tfidf)
acc10,f110 = random_forest.random_forest_with_majority_votes(train_data_x_tfidf, train_data_y_tfidf, dev_data_x_tfidf, dev_data_y_tfidf, dev_data_user_tfidf,
                                                                       test_data_x_tfidf,test_data_user_tfidf)
acc11,f111 = random_forest.random_forest_max_average_prob(train_data_x_tfidf, train_data_y_tfidf, dev_data_x_tfidf, dev_data_y_tfidf, dev_data_user_tfidf,
                                                                       test_data_x_tfidf,test_data_user_tfidf)
print("0r_baseline accuracy_score is " + str(acc5))
print("0r_baseline f1_score is " + str(f15))

print("knn accuracy_score is " + str(acc4))
print("knn f1_score is " + str(f14))

print("native_bayes accuracy_score is " + str(acc1))
print("native_bayes f1_score is " + str(f11))
print("native_bayes_with_majority_votes accuracy_score is " + str(acc2))
print("native_bayes_with_majority_votes f1_score is " + str(f12))
print("native_bayes_with_max_average_prob accuracy_score is " + str(acc3))
print("native_bayes_with_max_average_prob f1_score is " + str(f13))

print("logistic_regression accuracy_score is " + str(acc6))
print("logistic_regression f1_score is " + str(f16))
print("logistic_regression_with_majority_votes accuracy_score is " + str(acc7))
print("logistic_regression_with_majority_votes f1_score is " + str(f17))
print("logistic_regression_with_max_average_prob accuracy_score is " + str(acc8))
print("logistic_regression_with_max_average_prob f1_score is " + str(f18))

print("random_forest accuracy_score is " + str(acc9))
print("random_forest f1_score is " + str(f19))
print("random_forest_with_majority_votes accuracy_score is " + str(acc10))
print("random_forest_with_majority_votes f1_score is " + str(f110))
print("random_forest_with_max_average_prob accuracy_score is " + str(acc11))
print("random_forest_with_max_average_prob f1_score is " + str(f111))
