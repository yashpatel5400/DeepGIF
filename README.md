# DeepGIF
Video style transfer using convolutional networks, with tracking and masks for GIFs!!! Implemented for Computer Vision 2016 (Princeton University) by Richard Du, Yash Patel, and Jason Shi

![](https://media.giphy.com/media/l3q2GFXj10Zk4u6k0/source.gif =250x)
![](https://media.giphy.com/media/l3q2PsoY9acvLFELS/source.gif =250x)
![](https://media.giphy.com/media/26xBP8Jg8b5tPHO9i/source.gif =250x)
![](https://media.giphy.com/media/l3q2Zr1IYQyT6Jz3y/giphy.gif =250x)

### Requirement ###

- Python >=3.4
  - TensorFlow 0.12.0

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
