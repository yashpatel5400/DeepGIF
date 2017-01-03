# DeepArch
Video style transfer using convolutional networks, with tracking and masks. Implemented for Computer Vision 2016 (Princeton University) by Richard Du, Yash Patel, and Jason Shi

### Requirement ###

- Python >=3.4
  - TensorFlow 0.11.0
- Node >=6.9


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