"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
from . import facenet
from . import detect_face
import random
from time import sleep

def main(pnet,rnet,onet,path):
    #sleep(random.random())

    # Store some git revision info in a text file in the log directory
    #facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))


    minsize = 20 # minimum size of face
    threshold = [ 0.7, 0.8, 0.8 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    #bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_1sec.txt')


    nrof_successfully_aligned = 0

    image_path = path
    try:
        img = misc.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
    if img.ndim<2:
        return
    elif img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:,:,0:3]
    bounding_boxes, _ = detect_face.detect_face(img, minsize,pnet,rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    image = []
    bounding_boxex = []
    if nrof_faces>0:
        for i in range(nrof_faces) :
            det = bounding_boxes[i,0:4]
            img_size = np.asarray(img.shape)[0:2]
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0],0) #left
            bb[1] = np.maximum(det[1],0) #top
            bb[2] = np.minimum(det[2],img_size[1]) #right
            bb[3] = np.minimum(det[3],img_size[0]) #bottom
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(cropped, (160,160), interp='bilinear')
            bounding_boxex.append([bb[0],bb[1],bb[2],bb[3]])
            image.append(scaled)
                    #text_file.write('%s\n' % (output_filename))
    return bounding_boxex,image
