
from keras.models import model_from_json
import time
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
import pandas as pd
import os
import cv2
from datetime import datetime
import shutil
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import time

from AnalysisModule.config import GPU_MEM_FRACTION

class Classification:
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        MODEL_NAME = 'cr_nor_pat_pot_line_19-0.9149.hdf5'
        MODEL_JSON_FILE_NAME = 'cr_nor_pat_pot_line_19-0.9149.json'

        self.FULLSIZE_IMAGE_PATH = os.path.join("/workspace/Modules/classification/", 'images')
        self.SLICE_IMAGE_PATH = os.path.join(self.FULLSIZE_IMAGE_PATH, 'slices')
        self.PATCH_SIZE = 256
        self.INPUT_IMAGE_SIZE = 224
        self.MORE_CONTEXT = 27
        self.CLASS_MODE = 'categorical'  # 'binary' # 'categorical'
        self.ACTIVATION = 'softmax'  # 'sigmoid' # 'softmax'
        self.LOSS = 'categorical_crossentropy'  # 'binary_crossentropy' # 'categorical_crossentropy'
        self.OPTIMIZER = 'Adam'
        self.METRICS = ['accuracy']
        self.BATCH_SIZE = 1

        #
        json_file = open(os.path.join(self.path, MODEL_JSON_FILE_NAME), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
        set_session(tf.Session(config=config))
        
        # self.model = None
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(os.path.join(self.path, MODEL_NAME))
        self.model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER, metrics=self.METRICS)
        self.model._make_predict_function()


    def make_dir(self, dir_name):
        try:
            if not (os.path.isdir(dir_name)):
                os.makedirs(os.path.join(dir_name))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise

    def hangul_filepath_imread(self, filePath):
        stream = open(filePath.encode("utf-8"), "rb")
        bytes = bytearray(stream.read())
        numpyArray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)


    def coordinate_checker(self, width, height, ymin, ymax, xmin, xmax):
        if ymin < 0:
            ymin = 0
        if ymax > height:
            ymax = height
        if xmin < 0:
            xmin = 0
        if xmax > width:
            xmax = width
        return ymin, ymax, xmin, xmax

    def inference_by_path(self, image_path):
        total_start = time.time()

        tmp = image_path.split("/")
        image_name = tmp[len(tmp)-1]

        start = time.time()
        slices_dir_name = image_name.split(".")[0] + str(datetime.now().timestamp()).replace('.', '')
        slices_dir = os.path.join(self.SLICE_IMAGE_PATH, slices_dir_name)
        self.make_dir(self.SLICE_IMAGE_PATH)
        self.make_dir(slices_dir)
        self.make_dir(os.path.join(slices_dir, "slices"))
        end = time.time()
        print("Make slice directory time : {}".format(end - start))

        start = time.time()
        image = self.hangul_filepath_imread(image_path)
        height, width = image.shape

        for y in range(0, height - self.PATCH_SIZE, self.PATCH_SIZE):
            for x in range(0, width - self.PATCH_SIZE, self.PATCH_SIZE):
                # slice as 310x310
                xmin = x - self.MORE_CONTEXT
                xmax = x + self.PATCH_SIZE + self.MORE_CONTEXT
                ymin = y - self.MORE_CONTEXT
                ymax = y + self.PATCH_SIZE + self.MORE_CONTEXT
                ymin, ymax, xmin, xmax = self.coordinate_checker(width, height, ymin, ymax, xmin, xmax)
                crop = image[ymin:ymax, xmin:xmax]
                cv2.imwrite(os.path.join(slices_dir, "slices", '_' + str(x) + '_' + str(y) + '.jpg'), crop)



        data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_generator = data_generator.flow_from_directory(slices_dir,
                                                            target_size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
                                                            color_mode='rgb',
                                                            batch_size=self.BATCH_SIZE,
                                                            class_mode=None,
                                                            shuffle=False,
                                                            seed=42)

        step_size_test = test_generator.n // test_generator.batch_size

        test_generator.reset()
        pred = self.model.predict_generator(test_generator,
                                              steps=step_size_test,
                                              verbose=1)
        end = time.time()
        print("inference images of patch time : {}".format(end - start))

        filenames = test_generator.filenames
        results = []


        if self.CLASS_MODE == 'categorical':
            for k in range(0, len(pred)):
                result = np.where(pred[k] == np.amax(pred[k]))
                if result[0][0] != 1: # save result if not normal
                    patch_info = filenames[k].split('_')
                    xmin = int(patch_info[-2])
                    ymin = int(patch_info[-1].split(".")[0])
                    result = {
                        "label" : [
                          # {'NR': 1, 'PAT': 2, 'WHITE': 4, 'POT': 3, 'CR': 0}
                            { 'description':'crack', 'score': pred[k][0] },
                            { 'description': 'normal', 'score': pred[k][1] },
                            { 'description': 'patch', 'score': pred[k][2] },
                            { 'description': 'pothole', 'score': pred[k][3] },
                            { 'description': 'line', 'score': pred[k][4] }
                        ],
                        "position" : {
                            'x': xmin,
                            'y': ymin,
                            'w': self.PATCH_SIZE,
                            'h': self.PATCH_SIZE
                        }
                    }
                    results.append(result)

        self.result = results
        shutil.rmtree(slices_dir)
        total_end = time.time()
        print("classfication fin {}".format(total_end - total_start))

        return self.result

