import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import L1W4 as dl


def read_data():
    data = pd.read_csv('../data/heart.csv')
    return data


def data_normalization(data):
    data = data.drop(['target'], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data


def split_data(data):
    X = data
    y = pd.read_csv('../data/heart.csv')['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = np.array(X_train.T)
    X_test = np.array(X_test.T)
    y_train = np.array(y_train).reshape(1, len(y_train))
    y_test = np.array(y_test).reshape(1, len(y_test))
    return X_train, X_test, y_train, y_test


def main():
    data = read_data()
    data = data_normalization(data)
    X_train, X_test, y_train, y_test = split_data(data)
    model = dl.L_layer_model(X_train, y_train, layers_dims=[13, 10, 5, 1], learning_rate=0.05, num_iterations=20000,
                             print_cost=True)
    score_train = dl.score(X_train, y_train, model)
    score_test = dl.score(X_test, y_test, model)
    print('Train accuracy: ', score_train)
    print('Test accuracy: ', score_test)
    pre_data = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
    predictions = dl.predict(pre_data, model)
    if predictions == 0:
        print('The patient is not likely to have a heart disease.')
    else:
        print('The patient is likely to have a heart disease.')


if __name__ == '__main__':
    main()
