#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC


df = pd.read_csv('data/situacao_cliente.csv')
X_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]
Y_df = df['situacao']

X_dummies_df = pd.get_dummies(X_df).astype(int)
Y_dummies_df = Y_df

X = X_dummies_df.values
Y = Y_dummies_df.values

TRAINNING_RATE = 0.8
TEST_RATE = 0.1
total_elements = len(Y)

trainning_size = int(total_elements * TRAINNING_RATE)
test_size = int(total_elements * TEST_RATE)
validation_size = total_elements - trainning_size - test_size

X_trainning = X[:trainning_size]
Y_trainning = Y[:trainning_size]

test_limit = trainning_size + test_size

X_test = X[trainning_size:test_limit]
Y_test = Y[trainning_size:test_limit]

X_validation = X[test_limit:]
Y_validation = Y[test_limit:]

# data for base analysis algorithm
base_total_matches = max(Counter(Y_test).values())
base_matches_rate = 100.0 * base_total_matches / test_size

print "Taxa de acertos base: %f %%" % base_matches_rate


def fit_and_predict(model, algoritm_name, X_trainning, Y_trainning, X_test, Y_test):
    model.fit(X_trainning, Y_trainning)

    result = model.predict(X_test)

    matches = result == Y_test

    total_matches = sum(matches)
    total_elements_test = len(Y_test)

    print "Taxa de acertos do algoritmo {0}: {1} ".format(algoritm_name, round(100.0 * total_matches / total_elements_test, 2))
    return total_matches


def validate_algoritm(algorithm, X_validation, Y_validation):
    """
        Com o melhor algoritmo, roda o fit and predict sobre os dados de validacao (validation_data)
    """
    algorithm.fit(X_validation, Y_validation)

    validation_result = algorithm.predict(X_validation)

    matches = validation_result == Y_validation

    total_matches = sum(matches)
    total_elements_test = len(Y_validation)

    print "Total de elementos de validação", total_elements_test

    print "Taxa de acertos do melhor algoritmo: {0} %".format(round(100.0 * total_matches / total_elements_test, 2))


multinomialModel = MultinomialNB()
adaBoostModel = AdaBoostClassifier()
oneVsRestModel = OneVsRestClassifier(LinearSVC(random_state=0))
oneVsOneModel = OneVsOneClassifier(LinearSVC(random_state=0))

total_matches_multinomial = fit_and_predict(multinomialModel, "MultinomialNB",
                                            X_trainning, Y_trainning, X_test, Y_test)

total_matches_adaboost = fit_and_predict(adaBoostModel, "AdaBoost",
                                         X_trainning, Y_trainning, X_test, Y_test)

total_matches_one_vs_rest = fit_and_predict(oneVsRestModel, "OneVsRest",
                                            X_trainning, Y_trainning, X_test, Y_test)

total_matches_one_vs_one = fit_and_predict(oneVsOneModel, "OneVsOne",
                                           X_trainning, Y_trainning, X_test, Y_test)


print "Total de elementos analisados no teste: {0}".format(len(Y_test))


results_map = {}
results_map[total_matches_multinomial] = multinomialModel
results_map[total_matches_adaboost] = adaBoostModel
results_map[total_matches_one_vs_rest] = oneVsRestModel
results_map[total_matches_one_vs_one] = oneVsOneModel

best_algorithm_key = max(results_map)
best_algorithm = results_map[best_algorithm_key]

print "Melhor algoritmo:", type(best_algorithm)

validate_algoritm(best_algorithm, X_validation, Y_validation)
