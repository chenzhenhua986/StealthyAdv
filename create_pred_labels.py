#---------------------------------------------------------
#Create evaluation images with label ids for the cityscapes dataset

# Adapted from: Fully Convolutional Networks for Semantic Segmentation by Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015 and PAMI 2016. http://fcn.berkeleyvision.org


#Parameters to be set by the user:

python_caffe_root='/l/vision/v5/chen478/Fall2018/caffe/python'
fcn_net='/l/vision/v5/chen478/Fall2018/fcn.cityscapes/train/cityscapes-fcn8s-2x.caffemodel'	#the caffe model to be evaluated
path_imgs='/l/vision/v5/chen478/Fall2018/mapillary/testing/images/'	#folder and subfolders will be searched
path_result='/l/vision/v5/chen478/Fall2018/fcn.mapillary/results/gt_test/'     #folder where label images will be saved
pixel_mean=(71.60167789, 82.09696889, 72.30608881)      #mean pixel from the dataset the model was trained on

#---------------------------------------------------------


import sys
sys.path.insert(0, python_caffe_root)

from PIL import Image
import caffe
import numpy as np
import os
import setproctitle

setproctitle.setproctitle(os.path.basename(os.getcwd()))

# init
net = caffe.Net('deploy_8s.prototxt', fcn_net, caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(0)

eval_images=[]

#find all images
for f in os.listdir(path_imgs):
    root_label="/l/vision/v5/chen478/Fall2018/fcn.mapillary/results/seg_res_test"
    gt_label_path='/l/vision/v5/chen478/Fall2018/fcn.mapillary/results/gt_test/'
    eval_images.append((path_imgs + f, root_label+'/'+f, gt_label_path + f))

n_images=len(eval_images)

#create label images
for idx in range(n_images):
    print(eval_images[idx][1])
    im = Image.open(eval_images[idx][0])
    im = im.resize((400, 200), Image.ANTIALIAS)
    #im = im.resize((500, 500), Image.ANTIALIAS)
    width, height = im.size
    #print(width)
    #print(height)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array(pixel_mean)
    in_ = in_.transpose((2,0,1))
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    out_np=np.array(out, dtype=np.uint8)
    #print(out_np.shape)
    #for i in range(height):
    # for j in range(width):
    #   if out_np[i, j] == 11 or out_np[i, j] == 12:
    #     out_np[i, j] = 0

    masked_im = Image.fromarray(out_np)
    masked_im.save(path_result + eval_images[idx][2].split('/')[9])
