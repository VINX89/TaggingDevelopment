#! /usr/bin/env python
# coding: utf-8 

"""
@file    calib_models.py
@author  Kevin Heinicke, Vincenzo Battista
@date    17/09/2017

@brief   Collection of models useful to calibrate a classifier output
"""

from sklearn.linear_model import LogisticRegression
import numpy as np

class PolynomialLogisticRegression:
    """
    Class implementing a polynomial, logistic model.
    The polynomial can be a function of any power.
    The value of the polynomial is converted to a probability (between 0 and 1)
    via a 'link' function (in this case, a logistic function)
    """

    def __init__(self, power, *args, **kwargs):
        self.power = power
        self.lr = LogisticRegression(*args, **kwargs)

    def transform(self, X):
        Xnew = []
        for i in range(1, self.power+1):
            Xnew.append(X**i)
        Xnew = np.column_stack(Xnew)
        return Xnew

    def fit(self, X, y, sample_weight=None):
        Xnew = self.transform(X)
        self.lr.fit(Xnew, y, sample_weight)
        
    def fit_transform(self, X, y=None, **fit_params):
        Xnew = self.transform(X)
        return self.lr.fit_transform(Xnew, y, **fit_params)

    def predict(self, X):
        Xnew = self.transform(X)
        return self.lr.predict(Xnew)

    def predict_proba(self, X):
        Xnew = self.transform(X)
        return self.lr.predict_proba(Xnew)

    def score(self, X):
        Xnew = self.transform(X)
        return self.lr.score(Xnew)
