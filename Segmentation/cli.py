# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import webbrowser
import subprocess

import learner
from N4 import default_N4

def train(dataset):
    """
    Train an N4 models to predict affinities
    """
    data_folder = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset + '/'
    data_provider = DPTransformer(data_folder, 'train.spec')

    learner.train(default_N4(), data_provider, data_folder, n_iterations=10000)

def predict():
    """
    Realods a model previously trained
    """
    data_folder = './'
    learner.predict(default_N4(), data_folder, "kiwi.jpg")

if __name__ == '__main__':
    predict()
