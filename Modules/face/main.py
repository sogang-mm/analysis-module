import pickle
import os
import dlib
from .face_py import align_dlib
import numpy as np
import tensorflow as tf
from .face_py import facenet
from .face_py import detect_face
from .face_py import align_dataset_mtcnn
import sys
import cv2
import math
import collections
class Face():
    def __init__(self) :
        classifier_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),'classifier.pkl')

        with open(classifier_filename, 'rb') as infile:
            (self.models,self.class_names) = pickle.load(infile)

        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.25
            self.sess_detect = tf.Session(config = config)
            model =os.path.join(os.path.dirname(os.path.realpath(__file__)),'20180402-114759.pb')
            facenet.load_model(model)
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.get_shape()[1]
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
            sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess1.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess1, None)

    def inference_by_path(self,image_path):
        result = []
        input_path = image_path
        (bounding_boxes, images) = align_dataset_mtcnn.main(self.pnet,self.rnet,self.onet,input_path)
        if not bounding_boxes :
            return result
        batch_size = 20
        face_images = []
        face_location = []
        frame_name = []
        for i in range(len(images)):
            images[i] = cv2.resize(images[i],(160,160),interpolation=cv2.INTER_LINEAR)
            face_images.append(images[i])
        nrof_images = len(face_images)
        faces = np.zeros((nrof_images,160,160,3))
        for i in range(nrof_images) :
            if faces[i].ndim == 2:
                faces[i] = facenet.to_rgb(faces[i])
            img = facenet.prewhiten(face_images[i])
            faces[i,:,:,:] = img
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, self.embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            face_index = faces[start_index:end_index,:,:,:]
            feed_dict = {self.images_placeholder: face_index, self.phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = self.sess_detect.run(self.embeddings, feed_dict=feed_dict)
        predictions = self.models.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)

        #best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        bounding_tmp = []
        for i in range(len(bounding_boxes)):
            bounding_tmp.append([bounding_boxes[i][0],bounding_boxes[i][1],bounding_boxes[i][2]-bounding_boxes[i][0],bounding_boxes[i][3]-bounding_boxes[i][1]])
        tmp = predictions.copy()
        for i in range(len(best_class_indices)) :
            tmp[i].sort()
            mklist = tmp[i][::-1]
            #if (best_class_probabilities[i] > 0.02):
            #    dic = {self.class_names[best_class_indices[i]]:best_class_probabilities[i]}
            #    result_tmp = [bounding_tmp[i],dic]
            #    result.append(result_tmp)
            dic={self.class_names[list(predictions[i]).index(mklist[0])]:mklist[0],self.class_names[list(predictions[i]).index(mklist[1])]:mklist[1],self.class_names[list(predictions[i]).index(mklist[2])]:mklist[2],self.class_names[list(predictions[i]).index(mklist[3])]:mklist[3],self.class_names[list(predictions[i]).index(mklist[4])]:mklist[4]}
            result_tmp = [bounding_tmp[i],dic]
            result.append(result_tmp)
        return result
