print('Probiha importovani potrebnych knihoven, tato operace muze trvat i nekolik minut.')
print('\n')

import os
from datetime import datetime
import csv

import yaml

# load configuration file
config_path = 'config.yaml'
with open(config_path, 'rb') as f:
    config = yaml.load(f, Loader=yaml.Loader)

# turn off tensorflow warning informations if required
if not config['DEBUG_MODE']:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scales_counter import ScalesCounter

import tensorflow as tf
import cv2
from tqdm import tqdm

from skimage.io import imsave


if __name__ == "__main__":
    print('******************************************')
    print('* Program ScalesCounter uspesne spusten! *')
    print('*                Vitejte!                *')
    print('******************************************')
    print('')
    print('Program muzete kdykoliv ukoncit stisknutim klaves CTRL+C v tomto okne.')
    print('Vysledky jsou ukladany prubezne a behem chodu programu se na ne muzete divat.')
    print('')


    # get input_images_path
    current_path = os.getcwd()
    root_path = os.path.dirname(current_path)
    input_images_path = os.path.join(root_path, config['IMAGES_INPUT_FOLDER'])

    # check if input_image_path is a directory
    if not os.path.isdir(input_images_path):
        print('Slozka se vstupnimi obrazky \'images_input\' ve slozce vedle programu neexistuje!')
        print('\n')
        os.system('pause')

    # save all input images paths
    list_of_images_paths = []
    for root, _, files in os.walk(input_images_path, topdown=False):
        for name in files:
            if (name[-4:] == '.PSD' or name[-4:] == '.JPG' or name[-4:] == '.CR2' or
                    name[-4:] == '.psd' or name[-4:] == '.jpg' or name[-4:] == '.png'):
                input_image = os.path.join(root, name)
                output_image = os.path.split(input_image.replace(
                    config['IMAGES_INPUT_FOLDER'], config['IMAGES_OUTPUT_FOLDER'], 1))[0]
                list_of_images_paths.append([input_image, output_image])

    print('Nalezeno', len(list_of_images_paths), 'fotek ke zpracovani.')
    print('')

    # make output folders if requiered
    if config['IMAGE_OUTPUT']:
        list_of_folders = []
        for path in list_of_images_paths:
            if path[1] not in list_of_folders:
                list_of_folders.append(path[1])

        # sort list so that folders are makable
        list_of_folders.sort(key=len)

        for folder in list_of_folders:
            try:
                os.mkdir(folder)
            except FileExistsError:
                pass
        print('V souboru config.yaml jste zapli vizualizaci vystupu, slozky pro ne jsou nyni vytvorene.')
        print('')
    else:
        print('V souboru config.yaml jste vypli vizualizaci vystupu, pro zapnuti zmente hodnotu IMAGE_OUTPUT na True a restartujte program.')
        print('')

    # load models outside of the loop to speed up whole process
    yolo_model = cv2.dnn.readNetFromDarknet(
        config['YOLO_CONFIG_FILE'], config['YOLO_WEIGHTS_FILE'])

    unet_first_model = tf.keras.models.load_model(
        config['FIRST_MODEL_PATH'])
    unet_second_model = tf.keras.models.load_model(
        config['SECOND_MODEL_PATH'])
    unet_third_model = tf.keras.models.load_model(
        config['THIRD_MODEL_PATH'])

    models = [unet_first_model, unet_second_model, unet_third_model]

    # make name for output txt file with timestamp to prevent overwritting
    current_time = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    txt_file_name = 'scales_counter_' + current_time + '.txt'

    try:
        os.mkdir(os.path.join(root_path, config['TXT_OUTPUT_FOLDER']))
    except FileExistsError:
        pass

    txt_file_path = os.path.join(
        root_path, config['TXT_OUTPUT_FOLDER'], txt_file_name)


    # loop over all images
    for image_path in tqdm(list_of_images_paths, total=len(list_of_images_paths)):
        image = ScalesCounter()
        cut_image = image.yolo_recognizing(image_path[0], yolo_model)

        if cut_image is None:
            txt_output = [image_path[0], 0, 0, image.error_info]
            continue

        result = image.u_net(cut_image, models)

        # make output images if requirered
        if config['IMAGE_OUTPUT']:
            output_image_name = os.path.split(
                image_path[0])[1].split('.', 1)[0] + '.png'
            output_image_path = os.path.join(image_path[1], output_image_name)
            imsave(output_image_path, result[2])

        txt_output = [image_path[0], result[0], result[1], image.error_info]

        # write results to txt output file
        with open(txt_file_path, 'a', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(txt_output)

    print('')
    print('Program uspesne vyhodnotil vsechny vstupy.')
    os.system('pause')