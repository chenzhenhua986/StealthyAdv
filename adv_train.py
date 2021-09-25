#---------------------------------------------------------
# Start training a model for the cityscapes dataset, initializing with the VGG 16-layer net

# Adapted from: Fully Convolutional Networks for Semantic Segmentation by Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015 and PAMI 2016. http://fcn.berkeleyvision.org


#Parameters to be set by the user:

python_caffe_root='/l/vision/v5/chen478/Fall2018/caffe/python'
fcn_root='/l/vision/v5/chen478/Fall2018/fcn.mapillary/'				#path containing cityscapes_layers.py and surgery.py
weights = '/l/vision/v5/chen478/Fall2018/fcn.cityscapes/train/cityscapes-fcn8s-2x.caffemodel'	#weights for initialization: VGG 16-layer net
n_steps=100000		#number of training iteration (one iteration = one random image from the training set)
final_model_name='adv_cityscapes-fcn8s-2x.caffemodel'

#---------------------------------------------------------


import sys

sys.path.insert(0, python_caffe_root)
sys.path.insert(0, fcn_root)

import caffe
import surgery
import os
import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))


# init
caffe.set_mode_gpu()
caffe.set_device(0)
#caffe.set_device(2)
#caffe.set_device(3)

# initialize with VGG net
solver = caffe.SGDSolver('adv_solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

#train
solver.step(n_steps)

solver.net.save(final_model_name)
