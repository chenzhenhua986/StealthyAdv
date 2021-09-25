#---------------------------------------------------------
#Python Layer for the cityscapes dataset

# Adapted from: Fully Convolutional Networks for Semantic Segmentation by Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015 and PAMI 2016. http://fcn.berkeleyvision.org


#---------------------------------------------------------

import caffe
import numpy as np
from PIL import Image
import os
from random import randint
import random


class MapillaryLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - cityscapes_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for SBDD semantic segmentation.

        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.

        example

        params = dict(cityscapes_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """

        # config
        params = eval(self.param_str)
        self.label_dir = params['label_dir']
        self.img_dir = params['img_dir']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        print(self.img_dir)
        print(self.label_dir)
        # two tops: data and label
        if len(top) != 3:
            raise Exception("Need to define three tops: data, predicted label, and fake_label")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels

	self.indices=[]
	path_imgs=self.label_dir;
	for f in os.listdir(path_imgs):
            #print(self.img_dir + f)
            #print(self.label_dir + f)
            self.indices.append((self.img_dir + f, self.label_dir + f))

        #print(self.indices)
        #exit()

        self.idx = []
        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        #self.data = self.load_image(self.indices)
        #self.label = self.load_label(self.indices[self.idx])
        self.idx = random.sample(range(0, len(self.indices) - 1), 5)

        x=[-40, -30, -20, -10, 0, 10, 20, 30, 40]        
        y=[-20, -15, -10, -5, 0, 5, 10, 15, 20] 
        x_idx = randint(0, 8)
        y_idx = randint(0, 8)

        self.data = self.load_image(self.idx, x[x_idx], y[y_idx])
        self.label = self.load_label(self.idx, x[x_idx], y[y_idx])
        self.fakelabel = self.load_fakelabel()
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)
        top[2].reshape(*self.fakelabel.shape)
        #print(self.idx)
        #print(self.indices[self.idx])


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        top[2].data[...] = self.fakelabel
        #print(top[1].data.shape)


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx, x_offset, y_offset):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        for i in range(1):
	  im = Image.open(self.indices[idx[i]][0])
          im = im.resize((400, 200), Image.ANTIALIAS)
          #x=[-40, -30, -20, -10, 0, 10, 20, 30, 40]        
          #y=[-20, -15, -10, -5, 0, 5, 10, 15, 20] 
          #x_idx = randint(0, 8)
          #y_idx = randint(0, 8)
          im = im.crop((40 + x_offset, 20 + y_offset, 360 + x_offset, 180 + y_offset)) # left, top, right, bottom

	  in_ = np.array(im, dtype=np.float32)
          #print(in_.shape)
          in_ = in_[:,:,::-1]
          in_ -= self.mean
          in_ = in_.transpose((2,0,1))
          in_ = in_[np.newaxis, ...]
          #in_ = np.zeros(in_.shape)
          if i==0:
            res = in_
          else:
            res = np.append(res, in_, axis=0)

        return np.append(res, res, axis=0)
      
	#im = Image.open(idx[0])
        #im = im.resize((400, 200), Image.ANTIALIAS)        
	#in_ = np.array(im, dtype=np.float32)
        #return np.append(in_, in_, axis=0)


    def load_fakelabel(self):
        """
        a fake figure, who has the same shape of the input image
        """
        for i in range(1):
	  im = Image.open('rsz.png').convert('L')
          #im = im.resize((320, 160), Image.ANTIALIAS)        
	  label = np.array(im, dtype=np.float32)
          label = label[np.newaxis, ...]
          if i==0:
            res = label
          else:
            res = np.append(res, label, axis=0)
        return np.append(res, res, axis=0)

    def load_label(self, idx, x_offset, y_offset):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        for i in range(1):
	  im = Image.open(self.indices[idx[i]][1])
          #print(self.indices[idx[i]][1])
	  #im = Image.open(self.indices[i][1])
          #im = im.resize((46, 21), Image.ANTIALIAS)        
          im = im.resize((400, 200), Image.ANTIALIAS)        
          #x=[-40, -30, -20, -10, 0, 10, 20, 30, 40]        
          #y=[-20, -15, -10, -5, 0, 5, 10, 15, 20] 
          #x_idx = randint(0, 8)
          #y_idx = randint(0, 8)
          #im = im.crop((40 + x[x_idx], 20 + y[y_idx], 360 + x[x_idx], 180 + y[y_idx])) # left, top, right, bottom
          im = im.crop((40 + x_offset, 20 + y_offset, 360 + x_offset, 180 + y_offset)) # left, top, right, bottom
	  label = np.array(im, dtype=np.uint8)
          label = label[np.newaxis, np.newaxis, ...]
          #label = label[np.newaxis, ...]
          #print(label[label>0])
          if i==0:
            res = label
          else:
            res = np.append(res, label, axis=0)
        return np.append(res, res, axis=0)
