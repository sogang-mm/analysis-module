import cPickle
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys, cv2
import argparse
import __init__paths
from fast_rcnn.config import cfg
import caffe
import selective_search
from fast_rcnn.test_xybb import im_detect
from utils.nms import nms
from utils.timer import Timer
from fast_rcnn.bbox_transform import bbox_voting
from datasets import ds_utils

class vis_det:
    def __init__(self):
        # TODO
        #   - initialize and load model here
	self.CLASSES = ('__background__',)
	self.WIND = (0,)
	self.classes = sio.loadmat(os.path.join(cfg.ROOT_DIR,'meta_det.mat'))
	self.synsets = sio.loadmat(os.path.join(cfg.ROOT_DIR,'meta_det.mat'))
	for i in xrange(200):
		self.CLASSES = self.CLASSES + (self.synsets['synsets'][0][i][2][0],)
		self.WIND = self.WIND + (self.synsets['synsets'][0][i][1][0],)
	self.NETS = {'resnet269': ('ResNet-GBD',
                  'ResNet-269-GBD_iter_180000.caffemodel')}
	
    	self.prototxt = os.path.join(cfg.ROOT_DIR,'..', 'ResNet-GBD', 'deploy_ResNet269_GBD.prototxt')
    	self.caffemodel = os.path.join(cfg.ROOT_DIR,'..', 'ResNet-GBD', 'models', self.NETS['resnet269'][1])
    
	if not os.path.isfile(self.caffemodel):
        	raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(self.caffemodel))

	caffe.set_mode_gpu()
	caffe.set_device(0)

	self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

	with open(os.path.join(cfg.ROOT_DIR,'..','evaluation','bbox_means.pkl'), 'rb') as f:
        	self.bbox_means = cPickle.load(f)
	with open(os.path.join(cfg.ROOT_DIR,'..','evaluation','bbox_stds.pkl'), 'rb') as f:
        	self.bbox_stds = cPickle.load(f)

    	self.net.params['bbox_pred_finetune'][0].data[...] = \
            self.net.params['bbox_pred_finetune'][0].data * self.bbox_stds[:, np.newaxis]
    	self.net.params['bbox_pred_finetune'][1].data[...] = \
            self.net.params['bbox_pred_finetune'][1].data * self.bbox_stds + self.bbox_means

	print ("model loaded")
        self.result = None

    def inference_by_path(self, image_path):
        # TODO
        #   - Inference using image path

	# Load the demo image
    	im = cv2.imread(image_path)
	
    	timer = Timer()
    	timer.tic()
	raw_data=selective_search.ss(im)
    	boxes = np.maximum(raw_data[:, 0:4], 0).astype(np.uint16)
    	# Remove duplicate boxes and very small boxes and then take top k
    	keep = ds_utils.unique_boxes(boxes)
    	boxes = boxes[keep, :]
    	keep = ds_utils.filter_small_boxes(boxes, 50)
    	boxes = boxes[keep, :]
    	boxes = boxes[:300, :]
    	num_boxes = boxes.shape[0]
    
    	scores_batch = np.zeros((300, 201), dtype=np.float32)
    	boxes_batch = np.zeros((300, 4*201), dtype=np.float32)
    	rois = np.tile(boxes[0, :], (300, 1))
    	rois[:num_boxes, :] = boxes

    	for j in xrange(2):
        	roi = rois[j*150:(j+1)*150, :]
        	score, box = im_detect(self.net, im, roi,0,0)
        	scores_batch[j*150:(j+1)*150, :] = score# [:,:,0,0]
        	boxes_batch[j*150:(j+1)*150, :] = box


    	scores = scores_batch[:num_boxes, :]
    	boxes = boxes_batch[:num_boxes, :]
    	timer.toc()
    	print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, num_boxes)

	# Visualize detections for each class
    	CONF_THRESH = 0.3
    	NMS_THRESH = 0.4

        result = []
    	for cls_ind, cls in enumerate(self.CLASSES[1:]):
		cls_ind +=1
		cls_scores = scores[:,cls_ind]
		cls_boxes = boxes[:, cls_ind*4:(cls_ind+1)*4]
		cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
		keep = nms(cls_dets, NMS_THRESH)
		cls_dets = cls_dets[keep, :]
        	idx = np.where(cls_dets[:, -1]  >= CONF_THRESH)[0]
	
        	if len(idx) == 0:
        		continue

        	for i in idx:
        		bbox = cls_dets[i, :4]
        		score = cls_dets[i, -1]
        		result.append([(round(bbox[0],3),round(bbox[1],3),round(bbox[2]-bbox[0],3),round(bbox[3]-bbox[1],3)),{cls.encode('ascii','ignore') : round(score,3)}])
		
        self.result = result
	return self.result
