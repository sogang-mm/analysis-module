
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

class Classification:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        DATA_NAME = 'muhanit_1018'  # 'ZEB_50k_256'
        MODEL_NAME = '25-0.9296.hdf5'
        MODEL_JSON_FILE_NAME = 'model.json'

        self.FULLSIZE_IMAGE_PATH = os.path.join("/workspace/Modules/classification/", 'images')
        self.SLICE_IMAGE_PATH = os.path.join(self.FULLSIZE_IMAGE_PATH, 'slices')
        self.PATCH_SIZE = 256
        self.INPUT_IMAGE_SIZE = 224
        self.CLASS_MODE = 'categorical'  # 'binary' # 'categorical'
        self.ACTIVATION = 'softmax'  # 'sigmoid' # 'softmax'
        self.LOSS = 'categorical_crossentropy'  # 'binary_crossentropy' # 'categorical_crossentropy'
        self.OPTIMIZER = 'Adam'
        self.METRICS = ['accuracy']
        self.BATCH_SIZE = 1
        #
        # json_file = open(os.path.join(self.path, MODEL_JSON_FILE_NAME), "r")
        # loaded_model_json = json_file.read()
        # json_file.close()
        #
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.3
        # config.gpu_options.visible_device_list = "0"
        # set_session(tf.Session(config=config))
        #
        # self.model = None
        # self.model.load_weights(os.path.join(self.path, MODEL_NAME))
        # self.model._make_predict_function()
        # self.model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER, metrics=self.METRICS)



    def inference_by_path(self, image_path):
        tmp = image_path.split("/")
        image_name = tmp[len(tmp)-1]
        # TODO
        #   - Inference using image path

        # TODO
        #   - initialize and load model here
        DATA_NAME = 'muhanit_1018'  # 'ZEB_50k_256'
        MODEL_NAME = '25-0.9296.hdf5'
        MODEL_JSON_FILE_NAME = 'model.json'

        json_file = open(os.path.join(self.path, MODEL_JSON_FILE_NAME), "r")
        loaded_model_json = json_file.read()
        json_file.close()

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))

        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(self.path, MODEL_NAME))
        model._make_predict_function()
        model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER, metrics=self.METRICS)

        print(image_path)
        slices_dir_name = image_name.split(".")[0] + str(datetime.now().timestamp()).replace('.', '')
        slices_dir = os.path.join(self.SLICE_IMAGE_PATH, slices_dir_name)
        print("slices_dir", slices_dir)
        self.make_dir(self.SLICE_IMAGE_PATH)
        self.make_dir(slices_dir)
        self.make_dir(os.path.join(slices_dir, "slices"))

        print("=====================test1======================")

        # Process image
        image = self.hangul_filepath_imread(image_path)
        height, width = image.shape

        for y in range(0, height - self.PATCH_SIZE, self.PATCH_SIZE):
            for x in range(0, width - self.PATCH_SIZE, self.PATCH_SIZE):
                # slice as 768x768
                xmin = x - self.PATCH_SIZE
                xmax = x + self.PATCH_SIZE * 2
                ymin = y - self.PATCH_SIZE
                ymax = y + self.PATCH_SIZE * 2
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

        print("test_generator.n", test_generator.n)

        print("slices_dir", slices_dir)
        print("=====================step_size_test fin======================")

        test_generator.reset()

        print("=====================test_generator.reset() fin======================")

        pred = model.predict_generator(test_generator,
                                              steps=step_size_test,
                                              verbose=1)

        print("=====================self.model.predict_generator fin======================")


        filenames = test_generator.filenames
        predictions = []

        results = []


        if self.CLASS_MODE == 'categorical':
            for k in range(0, len(pred)):
                result = np.where(pred[k] == np.amax(pred[k]))
                # print(result[0][0])
                predictions.append(result[0][0])
                if result[0][0] == 0:
                    patch_info = filenames[k].split('_')
                    xmin = int(patch_info[-2])
                    ymin = int(patch_info[-1].split(".")[0])
                    result = [(xmin, ymin, self.PATCH_SIZE, self.PATCH_SIZE), {'crack': 100}]
                    results.append(result)
        else:
            if k >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        self.result = results
        shutil.rmtree(slices_dir)
        return self.result

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
        # if over? just cut max position
        if ymin < 0:
            ymin = 0
        if ymax > height:
            ymax = height
        if xmin < 0:
            xmin = 0
        if xmax > width:
            xmax = width
        return ymin, ymax, xmin, xmax

#test = Classification()
#print(test.inference_by_path("/workspace/Modules/classification/sample.jpg"))
