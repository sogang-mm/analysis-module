from keras.models import model_from_json 
import time
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
import pandas as pd
import os 
from util import *
import cv2

DATA_NAME = 'muhanit_1018' #'ZEB_50k_256'
MODEL_PATH = os.path.join('./',DATA_NAME, 'model', './')
MODEL_NAME = '25-0.9296.hdf5'
MODEL_JSON_FILE_NAME = 'model.json'

FULLSIZE_IMAGE_PATH = os.path.join('/hdd', DATA_NAME, 'images')
SLICE_IMAGE_PATH = os.path.join('/hdd', DATA_NAME, 'slices')
# TEST_PATH = os.path.join('/hdd', DATA_NAME, 'test')
PATCH_SIZE = 256
INPUT_IMAGE_SIZE = 224
CLASS_MODE = 'categorical' #'binary' # 'categorical'
ACTIVATION = 'softmax'#'sigmoid' # 'softmax'
LOSS = 'categorical_crossentropy' #'binary_crossentropy' # 'categorical_crossentropy'
OPTIMIZER = 'Adam'
METRICS = [ 'accuracy', 
           # single_class_precision(0), single_class_recall(0),
           # single_class_precision(1), single_class_recall(1),
           # single_class_precision(2), single_class_recall(2),
           # single_class_precision(3), single_class_recall(3),
           # single_class_precision(4), single_class_recall(4),
           # single_class_precision(5), single_class_recall(5),
           # single_class_precision(3), single_class_recall(6)
           ]
BATCH_SIZE = 1

# Find images to slice, this is where you should input full-size(3704x10000) image
file_list = os.listdir(FULLSIZE_IMAGE_PATH)
image_list = [file for file in file_list if file.endswith(".jpg")]
make_dir(SLICE_IMAGE_PATH)
make_dir(os.path.join(SLICE_IMAGE_PATH, 'images'))
image_name_without_hangul_list = []

# Process image
slice_start_time = time.time()
for image_name in image_list:
  image = hangul_filepath_imread(os.path.join(FULLSIZE_IMAGE_PATH, image_name))
  height, width = image.shape
  # hangul processing
  image_name = image_name_without_hangul(image_name)
  image_name_without_hangul_list.append(image_name)

  for y in range(0, height-PATCH_SIZE, PATCH_SIZE):
    for x in range(0, width-PATCH_SIZE, PATCH_SIZE):
      # slice as 768x768
      xmin = x-PATCH_SIZE
      xmax = x+PATCH_SIZE*2
      ymin = y-PATCH_SIZE
      ymax = y+PATCH_SIZE*2
      ymin, ymax, xmin, xmax = coordinate_checker(width, height, ymin, ymax, xmin, xmax)
      crop = image[ymin:ymax, xmin:xmax]
      cv2.imwrite(os.path.join(SLICE_IMAGE_PATH, 'images', image_name + '_' + str(x) + '_' + str(y) + '.jpg'), crop)
  print('image ', image_name, ' is sliced')
slice_end_time = time.time() - slice_start_time
print('slice_end_time : ', slice_end_time)
print(image_name_without_hangul_list)
data_generator=ImageDataGenerator(preprocessing_function=preprocess_input) 
test_generator=data_generator.flow_from_directory(SLICE_IMAGE_PATH, 
                                                target_size=(INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE), 
                                                color_mode='rgb', 
                                                batch_size=BATCH_SIZE,
                                                class_mode=None,
                                                shuffle=False,
                                                seed=42)
step_size_test=test_generator.n // test_generator.batch_size

json_file = open(os.path.join(MODEL_PATH, MODEL_JSON_FILE_NAME), "r")
loaded_model_json = json_file.read() 
json_file.close()

model_load_start = time.time()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(os.path.join(MODEL_PATH, MODEL_NAME))
loaded_model._make_predict_function()
loaded_model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
model_load_end = time.time() - model_load_start
print ('model_load_end : ', model_load_end)

inference_start_time = time.time()
test_generator.reset()
print(test_generator.class_indices)

pred=loaded_model.predict_generator(test_generator,
                  steps=step_size_test,
                  verbose=1)
inference_end_time = time.time() - inference_start_time
print('inference_end_time : ', inference_end_time)
print('pred : ',pred[0])

filenames=test_generator.filenames
predictions = []
output_json_list = []
for image_name_without_hangul in image_name_without_hangul_list:
  output_json = {}
  output_json['file_name'] = image_name_without_hangul
  output_json['crack'] = []
  output_json_list.append(output_json)

if CLASS_MODE == 'categorical':
  for k in range(0, len(pred)):
    result = np.where(pred[k] == np.amax(pred[k]))
    # print(result[0][0])
    predictions.append(result[0][0])
    if result[0][0] == 0:
      patch_info = filenames[k].split('_')
      image_name = patch_info[0].split('/')[1]
      xmin = patch_info[-2]
      ymin = patch_info[-1]

      for j in range(0, len(output_json_list)):
        if output_json_list[j]['file_name'] == image_name:
          output_json_list[j]['crack'].append({'x':xmin, 'y':ymin})
else : 
# if binary classification problem
  if k >= 0.5 :
    predictions.append(1)
  else :
    predictions.append(0)


#Make csv
# results=pd.DataFrame({"Filename":filenames,
#                       "Predictions":predictions})
# results.to_csv(os.path.join(DATA_NAME, 
#   MODEL_NAME.split('.')[0]+"_batch_" + str(BATCH_SIZE) +"output.csv"),index=False)

#Make json
import json
with open(os.path.join(DATA_NAME, 
  MODEL_NAME.split('.')[0]+"_batch_" + str(BATCH_SIZE) +"output.json"), 'w') as outfile:
  json.dump(output_json_list, outfile)

print('-----Summary------')
print('slice_end_time : ', slice_end_time)
print ('model_load_end : ', model_load_end)
print('inference_end_time : ', inference_end_time)

# code is from below
# #https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720