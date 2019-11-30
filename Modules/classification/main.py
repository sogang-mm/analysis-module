import base64

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
import json

from AnalysisModule import settings


class Classification:
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        # MODEL_NAME = '25-0.9296.hdf5'
        MODEL_NAME = 'actclc_27-0.9323.hdf5'
        MODEL_JSON_FILE_NAME = 'actclc_27-0.9323.json'

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
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
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

    def inference_by_path(self, seg_result, file_path):
        total_start = time.time()

        seg_img_path = os.path.join(self.path, settings.MEDIA_ROOT,
                                    str(datetime.now().timestamp()).replace('.', '') + ".png")
        with open(seg_img_path, 'wb') as seg_img:
            seg_img.write(base64.b64decode(seg_result))
        seg_img.close()
        tmp = seg_img_path.split("/")
        image_name = tmp[len(tmp) - 1]

        start = time.time()
        slices_dir_name = image_name.split(".")[0] + str(datetime.now().timestamp()).replace('.', '')
        slices_dir = os.path.join(self.SLICE_IMAGE_PATH, slices_dir_name)
        self.make_dir(self.SLICE_IMAGE_PATH)
        self.make_dir(slices_dir)
        self.make_dir(os.path.join(slices_dir, "slices"))
        end = time.time()
        print("Make slice directory time : {}".format(end - start))

        start = time.time()
        image = self.hangul_filepath_imread(seg_img_path)
        height, width, __ = image.shape
        json_file = open(file_path)
        json_array = json.load(json_file)

        cracks = json_array['results'][0]['module_result']

        # for y in range(0, height - self.PATCH_SIZE, self.PATCH_SIZE):
        #     for x in range(0, width - self.PATCH_SIZE, self.PATCH_SIZE):
        #         # slice as 310x310
        #         xmin = x - self.MORE_CONTEXT
        #         xmax = x + self.PATCH_SIZE + self.MORE_CONTEXT
        #         ymin = y - self.MORE_CONTEXT
        #         ymax = y + self.PATCH_SIZE + self.MORE_CONTEXT
        #         ymin, ymax, xmin, xmax = self.coordinate_checker(width, height, ymin, ymax, xmin, xmax)
        #         crop = image[ymin:ymax, xmin:xmax]
        #         cv2.imwrite(os.path.join(slices_dir, "slices", '_' + str(x) + '_' + str(y) + '.jpg'), crop)

        for j in range(0,len(cracks)):
            crack=cracks[j]
            labels = sorted(crack['label'], key=lambda label_list: (label_list['score']), reverse=True)
            if labels[0]['description'] == "crack" :
                x = int(crack['position']['x'])
                y = int(crack['position']['y'])
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
        # print(test_generator.class_indices)

        filenames = test_generator.filenames
        # predictions = []

        results = []

        if self.CLASS_MODE == 'categorical':
            for k in range(0, len(pred)):
                patch_info = filenames[k].split('_')
                xmin = int(patch_info[-2])
                ymin = int(patch_info[-1].split(".")[0])
                result = {
                    "label": [
                        {'description': 'ac', 'score': pred[k][0]},
                        {'description': 'lc', 'score': pred[k][1]},
                        # {'description': 'detail_norm', 'score': pred[k][3]},
                        {'description': 'tc', 'score': pred[k][2]}
                    ],
                    "position": {
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

# cls = Classification()
# cls.inference_by_path('/workspace/Modules/classification/sample.jpg')