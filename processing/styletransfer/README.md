# Style-Transfer Implementations 
## Standard Implementation
Implemented largely from scratch with Keras. This will take around 10-15 minutes on GPU compute time to complete, which means CPU completion is largely ruled out. The upside of this implementation is that it can be used with ANY style+content combination, meaning you're not limited to the ones you've trained on (unlike the faster implementation described below). Here are some sample outputs of the algorithm:

## Fast Chainer Implementation
Significantly faster than the standard implementation provided above (around 10000x faster). Only downside is that it must be used with pre-trained models. HUGE gratitude to: https://github.com/yusuketomoto/chainer-fast-neuralstyle for developing and opening up the source for this fast neural style implementation.

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/tubingen.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_1.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_1.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_2.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_2.jpg" height="200px">

### Requirement
- [Chainer](https://github.com/pfnet/chainer)
```
$ pip install chainer
```

- Pre-trained models (described more fully below)
```
https://github.com/gafr/chainer-fast-neuralstyle-models
```

### A collection of pre-trained models
Fashizzle Dizzle created pre-trained models collection repository, [chainer-fast-neuralstyle-models](https://github.com/gafr/chainer-fast-neuralstyle-models). Please download all the models you wish to use and insert them into the models folder. If this is not done manually, it will be prompted when trying to run.

### Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155)