## Holistically-Nested Edge Detection

### Introduction:

<img src="http://pages.ucsd.edu/~ztu/hed.jpg" width="400">

We use a new edge detection algorithm, holistically-nested edge detection (HED), fully described by: http://arxiv.org/abs/1504.06375. We owe these excellent results fully to: Saining Xie at UC San Diego

### Pretrained model

We provide the pretrained model and training/testing code for the edge detection framework Holistically-Nested Edge Detection (HED).
  0. Download the pretrained model (56MB) from (http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel) and place it in examples/hed/ folder (will be downloaded automatically when the code is run if not already).

### Installing 
 0. Install prerequisites for Caffe (http://caffe.berkeleyvision.org/installation.html#prequequisites)
 1. Caffe: install from https://github.com/BVLC/caffe

### Acknowledgment: 
This code is taken from a pretrained HED. Special thanks to:

  @misc{xie15hed,
      Author = {Xie, Saining and Tu, Zhuowen},
      Title = {Holistically-Nested Edge Detection},
      Booktitle = {Proceedings of IEEE International Conference on Computer Vision},
      Year  = {2015},
    }

This code is based on Caffe. Thanks to the contributors of Caffe. Thanks @shelhamer and @longjon for providing fundamental implementations that enable fully convolutional training/testing in Caffe.

    @misc{Jia13caffe,
      Author = {Yangqing Jia},
      Title = { {Caffe}: An Open Source Convolutional Architecture for Fast Feature Embedding},
      Year  = {2013},
      Howpublished = {\url{http://caffe.berkeleyvision.org/}}
    }

If you encounter any issue when using our code or model, please let me know.