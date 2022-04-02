import os
from datetime import datetime
import csv

import tensorflow as tf
import cv2
from tqdm import tqdm
from skimage.io import imsave

from scales_counter import ScalesCounter

def run(FINISH_UP, IMAGE_OUTPUT, INPUT_FOLDER, OUTPUT_FOLDER, YOLO_CONFIG_FILE, YOLO_WEIGHTS_FILE, FIRST_MODEL_PATH, SECOND_MODEL_PATH, THIRD_MODEL_PATH, TEMPORARY_IMAGE, CONFIDENCE_THRESHOLD, ENLARGE_BODY, CUT_SIZE, LOCAL_FILTER, MINIMUM_DISTANCE, TM, MAXIMUM_SCALES, MINIMUM_SCALES):
    print('******************************************')
    print('* Program ScalesCounter uspesne spusten! *')
    print('*                Vitejte!                *')
    print('******************************************')
    print('')
    print('Program muzete kdykoliv ukoncit stisknutim klaves CTRL+C v tomto okne.')
    print('Vysledky jsou ukladany prubezne a behem chodu programu se na ne muzete divat.')
    print('')

    # get input_images_path
    input_images_path = INPUT_FOLDER

    # check if input_image_path is a directory
    if not os.path.isdir(input_images_path):
        print('Slozka se vstupnimi obrazky \'images_input\'  neexistuje! Vytvorte ji podle instrukci.')
        print('\n')

    # chceck if last job was finished
    already_done = []
    if not FINISH_UP:
        # make name for outputs files with timestamp to prevent overwritting
        current_time = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        print('Job name:', current_time)
        print('\n')
    else:
        current_time = FINISH_UP
        print('Dokoncovani predchoziho behu s job name:', current_time)
        print('\n')
        # search for already done images
        with open(os.path.join(OUTPUT_FOLDER, current_time + '.txt')) as f:
            lines = f.readlines()

        for line in lines:
            for char_number in range(0, len(line)):
                if line[char_number] == '\t':
                    already_done.append(line[0:char_number])
                    break

    try:
        os.mkdir(os.path.join(OUTPUT_FOLDER))
    except FileExistsError:
        pass

    try:
        os.mkdir(os.path.join(OUTPUT_FOLDER, current_time))
    except FileExistsError:
        pass

    # save all input images paths
    list_of_images_paths = []
    for root, _, files in os.walk(input_images_path, topdown=False):
        for name in files:
            if (name[-4:] == '.PSD' or name[-4:] == '.JPG' or name[-4:] == '.CR2' or
                    name[-4:] == '.psd' or name[-4:] == '.jpg' or name[-4:] == '.png'):
                input_image = os.path.join(root, name)
                output_image = os.path.split(input_image.replace(
                    INPUT_FOLDER, os.path.join(OUTPUT_FOLDER, current_time), 1))[0]
                list_of_images_paths.append([input_image, output_image])

    print('Nalezeno', len(list_of_images_paths), 'fotek ke zpracovani.')
    print('')

    # make output folders if requiered
    if IMAGE_OUTPUT:
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
        print('Zapli jste vizualizaci vystupu, slozky pro ne jsou nyni vytvorene.')
        print('')
    else:
        print('Vypli jste vizualizaci vystupu, pro zapnuti zmente hodnotu IMAGE_OUTPUT na True a restartujte program.')
        print('')

    # load models outside of the loop to speed up whole process
    yolo_model = cv2.dnn.readNetFromDarknet(
        YOLO_CONFIG_FILE, YOLO_WEIGHTS_FILE)

    unet_first_model = tf.keras.models.load_model(
        FIRST_MODEL_PATH)
    unet_second_model = tf.keras.models.load_model(
        SECOND_MODEL_PATH)
    unet_third_model = tf.keras.models.load_model(
        THIRD_MODEL_PATH)

    models = [unet_first_model, unet_second_model, unet_third_model]

    # make txt output
    txt_file_name = current_time + '.txt'
    txt_file_path = os.path.join(OUTPUT_FOLDER, txt_file_name)


    # loop over all images
    for image_path in tqdm(list_of_images_paths, total=len(list_of_images_paths)):
        # check if image already done
        if image_path[0] in already_done:
            continue
    
        image = ScalesCounter(TEMPORARY_IMAGE, CONFIDENCE_THRESHOLD, ENLARGE_BODY, CUT_SIZE, MINIMUM_SCALES, MAXIMUM_SCALES, LOCAL_FILTER, TM, IMAGE_OUTPUT, MINIMUM_DISTANCE)
        cut_image = image.yolo_recognizing(image_path[0], yolo_model)

        if cut_image is None:
            txt_output = [image_path[0], 0, 0, image.error_info]
            continue

        result = image.u_net(cut_image, models)

        # make output images if requirered
        if IMAGE_OUTPUT:
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