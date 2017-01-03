"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Client for using the segmentation portion of the
program, including doing the segmentation and watershed on input images.
"""

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
    N4 = default_N4()
    N4.train()

@cli.command()
@click.argument('input_img')
def predict(input_img):
    """
    Realods a model previously trained
    """
    N4 = default_N4()
    N4.predict(input_img)

if __name__ == '__main__':
    cli()
