#---------------------------------------------------------
#Create evaluation images with label ids for the cityscapes dataset

# Adapted from: Fully Convolutional Networks for Semantic Segmentation by Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015 and PAMI 2016. http://fcn.berkeleyvision.org


#Parameters to be set by the user:

python_caffe_root='/l/vision/v5/chen478/Fall2018/caffe/python'
fcn_net='snapshot/label_vanish3__iter_5000.caffemodel'	#the caffe model to be evaluated
#fcn_net='snapshot/fake_label__iter_10000.caffemodel'	#the caffe model to be evaluated
#fcn_net='snapshot/label_displace__iter_35000.caffemodel'	#the caffe model to be evaluated
path_imgs='/l/vision/v5/chen478/Fall2018/mapillary/testing/images/'	#folder and subfolders will be searched
path_result='/l/vision/v5/chen478/Fall2018/fcn.mapillary/train/results/attacked_images_test/'     #folder where label images will be saved
pixel_mean=(71.60167789, 82.09696889, 72.30608881)      #mean pixel from the dataset the model was trained on
#---------------------------------------------------------


import sys
sys.path.insert(0, python_caffe_root)

from PIL import Image
import caffe
import numpy as np
import os
import setproctitle
import vis

setproctitle.setproctitle(os.path.basename(os.getcwd()))

# init
net = caffe.Net('adv_deploy3.prototxt', fcn_net, caffe.TEST)
#caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(3)

eval_images=[]

#find all images
for f in os.listdir(path_imgs):
    root_label="/l/vision/v5/chen478/Fall2018/fcn.mapillary/results/seg_res_test"
    gt_label_path='/l/vision/v5/chen478/Fall2018/fcn.mapillary/results/gt_test/'
    eval_images.append((path_imgs + f, root_label+'/'+f, gt_label_path + f))


n_images=len(eval_images)


#create label images
accuracy = 0.0
for idx in range(n_images):
    #print(eval_images[idx][0])
    #print(eval_images[idx][1])
    #print(eval_images[idx][2])
    im = Image.open(eval_images[idx][0])
    im = im.resize((400, 200), Image.ANTIALIAS)
    x_offset=0
    y_offset=0
    im = im.crop((40 + x_offset, 20 + y_offset, 360 + x_offset, 180 + y_offset)) # left, top, right, bottom
    #im = im.resize((500, 500), Image.ANTIALIAS)
    #width, height = im.size
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array(pixel_mean)
    in_ = in_.transpose((2,0,1))
    net.blobs['data1'].reshape(1, *in_.shape)
    net.blobs['data1'].data[...] = in_

    l = Image.open(eval_images[idx][1])
    l = l.resize((400, 200), Image.ANTIALIAS)        
    l = l.crop((40 + x_offset, 20 + y_offset, 360 + x_offset, 180 + y_offset)) # left, top, right, bottom
    label = np.array(l, dtype=np.uint8)
    label = label[np.newaxis, np.newaxis,  ...]
    net.blobs['label'].data[...] = label

    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    out_np=np.array(out, dtype=np.uint8)

    #palette = vis.make_palette(34)
    l=[[0, 7], [1,8], [2,11], [3,12], [4,13], [5,17], [6,19], [7,20], [8,21], [9,22], [10,23], [11,24], [12,25], [13,26], [14,27],
    [15,28], [16,31], [17,32], [18,33]]
    
    out_np_new=np.zeros_like(out_np)  #initialize with unlabeld
    for i in l:
       out_np_new[out_np==i[0]]=i[1]

    #out_np_new = out_np_new.resize((1000, 500), Image.ANTIALIAS)
    #Save resulting array as image
    #Image.fromarray(vis.color_seg(out_np_new, palette)).resize((1000, 500), Image.ANTIALIAS).save(path_result+'eval_labels_' +  eval_images[idx][2], 'PNG')
    gt_label = np.array(Image.open(eval_images[idx][2]).crop((40 + x_offset, 20 + y_offset, 360 + x_offset, 180 + y_offset)), dtype=np.uint8)
    accuracy += np.sum(gt_label == out_np) / (320.0 * 160.0) 
    print(np.sum(gt_label == out_np) / (320.0 * 160.0))
    #masked_im = Image.fromarray(vis.vis_seg(im, out_np_new, palette)).resize((400, 200), Image.ANTIALIAS)
    #ID='attacked_' + eval_images[idx][1].split('_')[0] + '_' + eval_images[idx][1].split('_')[1] + '_' + eval_images[idx][1].split('_')[2] + '.png'
    #masked_im.save(path_result+ID)

accuracy = accuracy / n_images
print("overall accuracy: " + str(accuracy))




