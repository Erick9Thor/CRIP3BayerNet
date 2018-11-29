import argparse

import pandas as pd
import numpy as np
from collections import OrderedDict
import stadistics
import math as m
import warnings
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def parseargsinput():
    parser = argparse.ArgumentParser(description='Bayers network', prog='Bayers network')
    parser.add_argument('-p', help='Path of data', required=True)
    parser.add_argument('-ho', help='Train HOLD-OUT', required=False)

    args = parser.parse_args()
    return vars(args)


def process_data(path):
    return pd.read_csv(path, sep=';').dropna()


def holdout(data, train_ratio=0.2):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(data.shape[0] * train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]

    x_train = data.iloc[indices_train].values
    x_val = data.iloc[indices_val].values

    print '*Muestras train:', len(x_train), ' (', train_ratio * 100, '%)'
    print '*Muestras validacion:', len(x_val), ' (', (1 - train_ratio) * 100, '%) \n'

    return x_train, x_val


def createTable(X_train, y_train):
    positives = 0
    negatives = 0
    table = OrderedDict()

    for i in range(X_train.size):
        for word in X_train[i].split():
            if word not in table:
                table[word] = [0, 0, 0]

            if y_train[i] == 1:
                table[word][1] += 1
                positives += 1
            else:
                table[word][2] += 1
                negatives += 1

            table[word][0] += 1

    return table, negatives, positives


def classify(tweet, table, positives, negatives):
    contador0 = 0
    contador1 = 0

    for word in tweet.split():
        if word in table:
            var_n = table[word][2]
            var_p = table[word][1]

            var1 = np.log(var_n / float(negatives))
            var2 = np.log(var_p / float(positives))

            contador0 += var1
            contador1 += var2

    if contador0 > contador1:
        return 0
    else:
        return 1

    # likelihood_pos = 0
    # likelihood_neg = 0
    #
    # n_words = len(table)
    #
    # for word in tweet.split():
    #     if word in table:
    #         likelihood_pos += m.log((table[word][1] + 1) / float(positives + 1 * n_words))
    #         likelihood_neg += m.log((table[word][2] + 1) / float(negatives + 1 * n_words))
    #
    #     else:
    #         likelihood_pos += m.log(1 / float(positives + 1 * n_words))
    #         likelihood_neg += m.log(1 / float(negatives + 1 * n_words))
    #
    # likelihood_pos += m.log(p_tweets / float(p_tweets + n_tweets))
    # likelihood_neg += m.log(n_tweets / float(p_tweets + n_tweets))
    #
    # if likelihood_neg > likelihood_pos:
    #     return 0
    # return 1


def mesure(predicted, real):
    print '-----------------------------------'
    print "Accuracy: ", accuracy_score(predicted, real)
    print "Precision:", precision_score(predicted, real)
    print "Recall:   ", recall_score(predicted, real)
    print "F1:", f1_score(predicted, real)
    print '-----------------------------------'


def main():
    warnings.filterwarnings("ignore")  # No mostrar por panatalla los warnings generados
    d = parseargsinput()

    print 'Leyendo datos...\n'
    dataset = process_data(d['p'])

    print 'Creando set de test y de train, metodo Holdout: ' + d['ho']
    train, test = holdout(dataset, float(d['ho']))

    print 'Creando tabla...'
    table, negatives, positives = createTable(train[:, 1], train[:, 3].astype(int))

    stadistics.printStatics(dataset, train, test, table, negatives, positives)

    # *******************************************************************
    # =============   Validacion   =====================================
    # *******************************************************************

    c_ok, c_fail = 0, 0
    predicted, real = [], []

    print '>>> Naive Bayes'
    print 'Validando conjunto de train...'

    y_valuestrain = train[:, 3]

    for i in range(len(train)):
        u = classify(train[i][1], table, positives, negatives)
        predicted.append(u)
        real.append(int(train[i][3]))
        if u == int(train[i][3]):
            c_ok += 1
        else:
            c_fail += 1

    print '----> Train <----'
    print 'Ok:', c_ok, '     Fail:', c_fail, '        (Total:', c_ok + c_fail, ')'
    print str((c_ok / float(c_ok + c_fail)) * 100), '%'
    mesure(predicted, real)

    print '\n\nValidando conjunto de test...'
    c_ok, c_fail = 0, 0
    predicted, real = [], []

    y_valuestest = test[:, 3]

    for i in range(len(test)):
        u = classify(test[i][1], table, positives, negatives)
        predicted.append(u)
        real.append(int(test[i][3]))
        if u == int(test[i][3]):
            c_ok += 1
        else:
            c_fail += 1

    print '----> Test <----'
    print 'Ok:', c_ok, '     Fail:', c_fail, '        (Total:', c_ok + c_fail, ')'
    print str((c_ok / float(c_ok + c_fail)) * 100), '%'
    mesure(predicted, real)
    print '\n'


if __name__ == '__main__':
    main()
