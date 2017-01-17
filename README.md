# DeepGIF
Video style transfer using convolutional networks, with tracking and masks for GIFs!

Implemented for Computer Vision 2016 (Princeton University) by Richard Du, Yash Patel, and Jason Shi. Feel free to make use of the files given here or contact us if anything is not working properly! Please note that the pre-trained models that are necessary are more fully described in their respective folder, i.e. in 'processing/styletransfer' and 'processing/segmentation.'

<img src="https://media.giphy.com/media/l3q2GFXj10Zk4u6k0/source.gif" width="400" height="400" />
<img src="https://media.giphy.com/media/l3q2PsoY9acvLFELS/source.gif" width="400" height="400" />
<img src="https://media.giphy.com/media/26xBP8Jg8b5tPHO9i/source.gif" width="400" height="400" />
<img src="https://media.giphy.com/media/l3q2Zr1IYQyT6Jz3y/giphy.gif" width="400" height="400" />

The algoirthm and necessary background information is fully laid out in the following paper: ["DeepGIF"](DeepGIF.pdf)

### Requirement ###

- Python = 2.7
  - TensorFlow 0.12.0
  - Keras
  - Chainer
  - Caffe

### How to run ###

    $ pip install -r requirements.txt
    $ gunicorn main:app --log-file=-


### Deploy to Heroku ###

    $ heroku apps:create [NAME]
    $ heroku buildpacks:add heroku/nodejs
    $ heroku buildpacks:add heroku/python
    $ git push heroku master

or Heroku Button.

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)
