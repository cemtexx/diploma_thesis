import pickle
import csv

from configure import *


# load pickle file
pickle_in = open(PICKLE_PATH, 'rb')
images_data = pickle.load(pickle_in)
pickle_in.close()

# transform dictionary from pickle to a data matrix for csv writter
header = [
    'image_name',
    'original_name',
    'W',
    'H',
    'body_centerX',
    'body_centerY',
    'square_lenght',
    'angle',
    'add_black',
    'body_is_centered',
    'is_randomly_oriented',
    'class1',
    'class2',
    'class1_CNN',
    'class2_CNN',
    'diff',
    'status'
]

body_is_centered = 0
is_randomly_oriented = 0

for image in images_data:
    images_data[image]['class1'] = len(images_data[image]['class1'])
    images_data[image]['class2'] = len(images_data[image]['class2'])
    images_data[image]['class1_CNN'] = len(images_data[image]['class1_CNN'])
    images_data[image]['class2_CNN'] = len(images_data[image]['class2_CNN'])
    images_data[image]['image_name'] = image

    for key in images_data[image]:

        if key == 'body_is_centered' and images_data[image][key] == False:
            body_is_centered += 1

        elif key == 'is_randomly_oriented' and images_data[image][key] == True:
            is_randomly_oriented += 1

evaluation = {
    'body_is_centered': body_is_centered,
    'is_randomly_oriented': is_randomly_oriented
}

evaluation_proc = {
    'body_is_centered': str(str(round((body_is_centered/len(images_data))*100, 3))) + ' %',
    'is_randomly_oriented': str(str(round((is_randomly_oriented/len(images_data))*100, 3))) + ' %'
}

try:
    with open(CSV_PATH, mode='w', newline='') as csv_file:
        fieldnames = header
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')

        writer.writeheader()

        for image in images_data:
            writer.writerow(images_data[image])

        writer.writerow(evaluation)
        writer.writerow(evaluation_proc)
except PermissionError:
    raise PermissionError('close excel before rewritting')

print('done')