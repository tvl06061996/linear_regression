import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
class LinearRegression(object):
    def __init__(self):
        self.W = np.array([])
             
    def train_test_data_split(self, X, y, test_size):
        X_number_of_test = math.ceil(X.shape[0] * test_size)
        y_number_of_test = math.ceil(y.shape[0] * test_size)
        X_train = X[X_number_of_test:]
        X_test = X[:X_number_of_test:]
        y_train = y[y_number_of_test:]
        y_test = y[:y_number_of_test]
        return X_train, X_test, y_train, y_test
    
    def cal_cost(self, h, y):
        m = len(y)     
        return (sum(h - y) ** 2) / (2 * m)

    def fit(self, X, y, alpha=0.00000001, interations=1000):
        self.W = np.array(np.ones(X.shape[1] + 1), dtype=np.int).T
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        m = len(y)
        cos_history = np.zeros(interations)
        for interation in range(interations):
            h = np.dot(X, self.W)
            loss = h - y
            gradient = sum(X.T.dot(loss)) / m
            self.W = self.W - alpha * gradient  
            cos_history[interation] = self.cal_cost(h, y)
        self.draw(cos_history)
    
    def draw(self, cost_history):
        fig,ax = plt.subplots(figsize=(12,8))
        ax.set_ylabel('J(Theta)')
        ax.set_xlabel('Iterations')
        _=ax.plot(range(1000), cost_history,'b.')
        plt.show()
        
    def coef_(self):
        return self.W
    
    def predict(self, X):
        W = self.W
        y_predicts = []
        for x in X:
            y_predict = W[0]
            y_predicts.append(y_predict + sum([W[i + 1] * x[i] for i in range(len(x))]))
        return y_predicts
    
    def score(self, y_true, y_pred):
        u = ((np.array(y_true) - np.array(y_pred)) ** 2).sum()
        v = ((np.array(y_true) - np.array(y_true).mean()) ** 2).sum()
        return 1 - u/v
  
