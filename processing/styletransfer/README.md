# Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
Used of : fast artistic style transfer by using feed forward network. HUGE gratitude to: https://github.com/yusuketomoto/chainer-fast-neuralstyle for developing and opening up the source for this fast neural style implementation.

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/tubingen.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_1.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_1.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_2.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_2.jpg" height="200px">

## Requirement
- [Chainer](https://github.com/pfnet/chainer)
```
$ pip install chainer
```
## A collection of pre-trained models
Fashizzle Dizzle created pre-trained models collection repository, [chainer-fast-neuralstyle-models](https://github.com/gafr/chainer-fast-neuralstyle-models). Please download all the models you wish to use and insert them into the models folder. If this is not done manually, it will be prompted when trying to run.

## License
MIT

## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155)