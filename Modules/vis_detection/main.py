import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


class vis_det:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
	self.CLASSES = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'window', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair_drier', 'tooth brush')

    	self.prototxt = os.path.join(cfg.MODELS_DIR,'..','coco','VGG16','faster_rcnn_end2end', 'test.prototxt')
    	self.caffemodel = os.path.join(cfg.DATA_DIR, "coco_vgg16_faster_rcnn_final.caffemodel")

	if not os.path.isfile(self.caffemodel):
        	raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(self.caffemodel))

	caffe.set_mode_gpu()
	caffe.set_device(0)
	cfg.TEST.HAS_RPN = True
	self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)

	print ("model loaded")
        self.result = None


    def inference_by_path(self, image_path):
        # TODO
        #   - Inference using image path
        im = cv2.imread(image_path)
	print image_path

        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
        CONF_THRESH = 0.5
        NMS_THRESH = 0.3
        result = []
        for cls_ind, cls in enumerate(self.CLASSES[1:]):
		cls_ind += 1 # because we skipped background
        	cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        	cls_scores = scores[:, cls_ind]
        	dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        	keep = nms(dets, NMS_THRESH)
        	dets = dets[keep, :]
		idx = np.where(dets[:, -1]  >= CONF_THRESH)[0]

        	for i in idx:
        		bbox = dets[i, :4]
        		score = dets[i, -1]
        		result.append([(round(bbox[0],3),round(bbox[1],3),round(bbox[2]-bbox[0],3),round(bbox[3]-bbox[1],3)),{cls.encode('ascii','ignore') : round(score,3)}])
		
        self.result = result
	return self.result
