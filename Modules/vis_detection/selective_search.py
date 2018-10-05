#!/usr/bin/env python
'''
Usage:
    ./ssearch.py input_image (f|q)
    f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''
 
import sys
import cv2
import numpy as np

def ss(im):
    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);
    flag=0
    if (im.shape[1] > 500) : 
	flag=1
	newWidth = 500
    	newHeight = int(im.shape[0]*500/im.shape[1])
	rat_w = float(im.shape[1])/newWidth
    	rat_h = float(im.shape[0])/newHeight	
	im = cv2.resize(im, (newWidth, newHeight)) 
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    ss.switchToSelectiveSearchQuality()
    # if argument is neither f nor q print help message
    
    # run selective search segmentation on input image
    try : 
    	rects = ss.process()
	print('Total Number of Region Proposals: {}'.format(len(rects)))
    	candidates=[]
    	for r in rects:
	    	x1, y1, w, h = np.uint16(r)
	    	x2=x1+w
	    	y2=y1+h
		if flag==1:
		    x1=int(x1*rat_w)
		    y1=int(y1*rat_h)
		    w=int(w*rat_w)
		    h=int(h*rat_h)
	    	    x2=x1+w
	    	    y2=y1+h
	    	box=np.array([x1,y1,x2,y2])
	    	candidates.append(box)
    	boxes=np.array(candidates)
	return boxes
    except : 
    	# resize image
	newWidth = 500
    	newHeight = 250
	rat_w = float(im.shape[1])/newWidth
    	rat_h = float(im.shape[0])/newHeight
    	im = cv2.resize(im, (newWidth, newHeight)) 
	ss.setBaseImage(im)
	ss.switchToSelectiveSearchQuality()	
	rects = ss.process()
   	print('Total Number of Region Proposals: {}'.format(len(rects)))

    	candidates=[]
    	for r in rects:
		    x1, y1, w, h = np.uint16(r)
		    x1=int(x1*rat_w)
		    y1=int(y1*rat_h)
		    w=int(w*rat_w)
		    h=int(h*rat_h)
		    x2=(x1+w)
		    y2=(y1+h)
		    box=np.array([x1,y1,x2,y2])
		    candidates.append(box)
    	boxes=np.array(candidates)	
    	return boxes

