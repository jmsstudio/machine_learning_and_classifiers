#!/usr/bin/python
#! -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

df_classifications = pd.read_csv('data/emails.csv')
texts = df_classifications['email']
Y_df = df_classifications['classificacao']

splitten_texts = texts.str.lower().str.split(' ')
words = set()

for word_list in splitten_texts:
    words.update(word_list)

word_count = len(words)
word_tuples = zip(words, xrange(word_count))

words_translator = {word: index for word, index in word_tuples}


def transform_text_to_array(text, translator):
    text_array = [0] * len(translator)

    for word in text:
        if word in translator:
            word_position = translator[word]
            text_array[word_position] += 1

    return text_array


phrases_array = [transform_text_to_array(
    text, words_translator) for text in splitten_texts]

X = np.array(phrases_array)
Y = np.array(Y_df.tolist())

TRAINNING_RATE = 0.8
total_elements = len(Y)

trainning_size = int(total_elements * TRAINNING_RATE)
validation_size = total_elements - trainning_size

X_trainning = X[:trainning_size]
Y_trainning = Y[:trainning_size]

X_validation = X[trainning_size:]
Y_validation = Y[trainning_size:]

# data for base analysis algorithm
base_total_matches = max(Counter(Y_trainning).values())
base_matches_rate = 100.0 * base_total_matches / trainning_size

print "Taxa de acertos base: %f %%" % base_matches_rate


def fit_and_predict(model, algoritm_name, X_trainning, Y_trainning):
    K = 10

    scores = cross_val_score(model, X_trainning, Y_trainning, cv=K)
    average = np.mean(scores)

    print "Taxa de acertos do algoritmo {0}: {1} %".format(algoritm_name, round(100.0 * average))
    return average


def validate_algoritm(algorithm, X_trainning, Y_trainning, X_validation, Y_validation):
    """
        Com o melhor algoritmo, roda o fit and predict sobre os dados de validacao (validation_data)
    """
    algorithm.fit(X_trainning, Y_trainning)

    validation_result = algorithm.predict(X_validation)

    matches = validation_result == Y_validation

    total_matches = sum(matches)
    total_elements_test = len(Y_validation)

    print "Total de elementos de validação", total_elements_test

    print "Taxa de acertos do melhor algoritmo com elementos de validação: {0} %".format(round(100.0 * total_matches / total_elements_test, 2))


multinomialModel = MultinomialNB()
adaBoostModel = AdaBoostClassifier()
oneVsRestModel = OneVsRestClassifier(LinearSVC(random_state=0))
oneVsOneModel = OneVsOneClassifier(LinearSVC(random_state=0))

total_matches_multinomial = fit_and_predict(multinomialModel, "MultinomialNB",
                                            X_trainning, Y_trainning)

total_matches_adaboost = fit_and_predict(adaBoostModel, "AdaBoost",
                                         X_trainning, Y_trainning)

total_matches_one_vs_rest = fit_and_predict(oneVsRestModel, "OneVsRest",
                                            X_trainning, Y_trainning)

total_matches_one_vs_one = fit_and_predict(oneVsOneModel, "OneVsOne",
                                           X_trainning, Y_trainning)


print "Total de elementos analisados no teste: {0}".format(len(Y_trainning))


results_map = {}
results_map[total_matches_multinomial] = multinomialModel
results_map[total_matches_adaboost] = adaBoostModel
results_map[total_matches_one_vs_rest] = oneVsRestModel
results_map[total_matches_one_vs_one] = oneVsOneModel

best_algorithm_key = max(results_map)
best_algorithm = results_map[best_algorithm_key]

print "Melhor algoritmo:", type(best_algorithm)

validate_algoritm(best_algorithm, X_trainning,
                  Y_trainning, X_validation, Y_validation)
