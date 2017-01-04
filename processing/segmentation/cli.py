"""
__author__ = Yash Patel, Richard Du, and Jason Shi
__description__ = Client for using the segmentation portion of the
program, including doing the segmentation and watershed on input images.
"""

import click
from model import segment_edges, download_model

@click.group()
def cli():
    pass

@cli.command()
def download():
    """
    Downloads the pretrained HED model
    """
    download_model()

@cli.command()
@click.argument('img')
def segment(img):
    """
    Reloads a model previously trained
    """
    segment_edges([img])

if __name__ == '__main__':
    cli()
