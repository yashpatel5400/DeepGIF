# -*- coding: utf-8 -*-
from __future__ import print_function

import click
import learner
from preprocess import preprocess_imgs

from model import default_N4, default_segnet

@click.group()
def cli():
    pass

@cli.command()
def preprocess():
    """
    Preprocess the BSD500 images for both the edges and full segmentation
    """
    preprocess_imgs()

@cli.command()
@click.argument('dataset', type=click.Choice(['edge', 'full']))
def train(dataset):
    """
    Train an N4 models to predict affinities
    """
    data_folder = '/' + dataset + '/'
    data_provider = DPTransformer(data_folder, 'train.spec')

    learner.train(default_N4(), data_provider, data_folder, n_iterations=10000)

@cli.command()
@click.argument('dataset', type=click.Choice(['edge', 'full']))
def predict():
    """
    Realods a model previously trained
    """
    data_folder = './'
    learner.predict(default_N4(), data_folder, "kiwi.jpg")

if __name__ == '__main__':
    cli()
