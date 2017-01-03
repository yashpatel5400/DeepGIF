## Holistically-Nested Edge Detection

Created by Saining Xie at UC San Diego

### Introduction:

<img src="http://pages.ucsd.edu/~ztu/hed.jpg" width="400">

We develop a new edge detection algorithm, holistically-nested edge detection (HED), which performs image-to-image prediction by means of a deep learning model that leverages fully convolutional neural networks and deeply-supervised nets.  HED automatically learns rich hierarchical representations (guided by deep supervision on side responses) that are important in order to resolve the challenging ambiguity in edge and object boundary detection. We significantly advance the state-of-the-art on the BSD500 dataset (ODS F-score of .790) and the NYU Depth dataset (ODS F-score of .746), and do so with an improved speed (0.4s per image). Detailed description of the system can be found in our [paper](http://arxiv.org/abs/1504.06375).

### Pretrained model

We provide the pretrained model and training/testing code for the edge detection framework Holistically-Nested Edge Detection (HED). Please see the Arxiv or ICCV paper for technical details. The pretrained model (fusion-output) gives ODS=.790 and OIS=.808 result on BSDS benchmark dataset.
  0. Download the pretrained model (56MB) from (http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel) and place it in examples/hed/ folder.

### Installing 
 0. Install prerequisites for Caffe(http://caffe.berkeleyvision.org/installation.html#prequequisites)
 0. Caffe: install from https://github.com/BVLC/caffe

### Acknowledgment: 
This code is taken from a pretrained HED. Special thanks to:

  @InProceedings{xie15hed,
      author = {"Xie, Saining and Tu, Zhuowen"},
      Title = {Holistically-Nested Edge Detection},
      Booktitle = "Proceedings of IEEE International Conference on Computer Vision",
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