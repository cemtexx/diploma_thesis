import os
import pickle

# turn off tensorflow warning informations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm import tqdm
from skimage.io import imread
import numpy as np
import tensorflow as tf

from configure import *
from centroid_counterpoint import counterpoint


pickle_in = open(PICKLE_PATH, 'rb')
images_data = pickle.load(pickle_in)
pickle_in.close()

dataset = []
for file_name in os.listdir(BODIES_PATH):
    if file_name[-4:] == '.png':
        dataset.append(file_name)

# sort
dataset = [str(i)+".png" for i in sorted([int(num.split('.')[0])
                                          for num in dataset])]

# load models
first_model = tf.keras.models.load_model(FIRST_MODEL_PATH)
second_model = tf.keras.models.load_model(SECOND_MODEL_PATH)
third_model = tf.keras.models.load_model(THIRD_MODEL_PATH)

for image in tqdm(dataset, total=len(dataset)):
    if images_data[image]['is_randomly_oriented']:
        images_data[image]['diff'] = None
        images_data[image]['status'] = 'NOK'
        continue

    image_path = os.path.join(BODIES_PATH, image)
    raw_image = imread(image_path)[:, :, :3]

    preds_model = first_model.predict(np.expand_dims(raw_image/255, axis=0))[0]
    results = counterpoint(preds_model, LOCAL_FILTER,
                           MINIMUM_DISTANCE, TM, raw_image)[:]

    #print(image, len(results[0][0]), len(results[0][1]), 'first')

    if len(results[0][0]) > MAXIMUM_SCALES or len(results[0][0]) < MINIMUM_SCALES or len(results[0][1]) > MAXIMUM_SCALES or len(results[0][1]) < MINIMUM_SCALES:
        second_preds_model = second_model.predict(
            np.expand_dims(raw_image/255, axis=0))[0]
        results = counterpoint(
            second_preds_model, LOCAL_FILTER, MINIMUM_DISTANCE, TM, raw_image)[:]
        #print(image, len(results[0][0]), len(results[0][1]), 'second')

        if len(results[0][0]) > MAXIMUM_SCALES or len(results[0][0]) < MINIMUM_SCALES or len(results[0][1]) > MAXIMUM_SCALES or len(results[0][1]) < MINIMUM_SCALES:
            third_preds_model = third_model.predict(
                np.expand_dims(raw_image/255, axis=0))[0]
            results = counterpoint(
                third_preds_model, LOCAL_FILTER, MINIMUM_DISTANCE, TM, raw_image)[:]
            #print(image, len(results[0][0]), len(results[0][1]), 'third')

            if len(results[0][0]) > MAXIMUM_SCALES or len(results[0][0]) < MINIMUM_SCALES or len(results[0][1]) > MAXIMUM_SCALES or len(results[0][1]) < MINIMUM_SCALES:
                images_data[image]['diff'] = None
                images_data[image]['status'] = 'NOK'
                #print(image, len(results[0][0]), len(results[0][1]), 'none')
                continue

    images_data[image]['class1_CNN'] = results[0][0]
    images_data[image]['class2_CNN'] = results[0][1]
    images_data[image]['diff'] = (abs(len(images_data[image]['class1_CNN']) - len(images_data[image]['class1'])) +
                                  abs(len(images_data[image]['class2_CNN']) - len(images_data[image]['class2'])))
    images_data[image]['status'] = 'OK'

pickle_out = open(PICKLE_PATH, "wb")
pickle.dump(images_data, pickle_out)
pickle_out.close()

print('done')