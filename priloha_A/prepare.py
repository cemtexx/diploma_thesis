import random
import os
import shutil
import pickle

from psd_tools import PSDImage
from tqdm import tqdm
import rawpy
import imageio
from PIL import Image

from configure import *


class Prepare():
    def __init__(self):
        try:
            os.mkdir(IMAGES_PATH)
        except FileExistsError:
            pass

        self.data = {}

    def randomize(self):
        print('prepare data')
        # makes a list of all files in a current folder
        files = []
        for file_name in os.listdir(RAW_IMAGES_PATH):
            files.append(file_name)
        
        # randomizes elements in a list
        random.shuffle(files)

        # renames all .png files in a random order by number
        for e, file in tqdm(enumerate(files), total=len(files)):
            if file[-4:] == '.png' or file[-4:] == '.PNG':
                from_dir = os.path.join(RAW_IMAGES_PATH, file)
                name = str(e) + '.png'
                target = os.path.join(IMAGES_PATH, name)
                shutil.copy2(from_dir, target)

                file_data = {'original_name': file}
                self.data[name] = file_data

            elif file[-4:] == '.jpg' or file[-4:] == '.JPG':
                self._jpg2png(file, e)

            elif file[-4:] == '.psd' or file[-4:] == '.PSD':
                self._psd2png(file, e)

            elif file[-4:] == '.CR2':
                self._raw2png(file, e)


        self._save_pickle()

    def _psd2png(self, file_name, e):
        from_dir = os.path.join(RAW_IMAGES_PATH, file_name)
        name = str(e) + '.png'
        save_path = os.path.join(IMAGES_PATH, name)

        psd = PSDImage.open(from_dir)
        psd.composite().save(save_path)

        file_data = {'original_name': file_name}
        self.data[name] = file_data

    def _raw2png(self, file_name, e):
        from_dir = os.path.join(RAW_IMAGES_PATH, file_name)
        name = str(e) + '.png'
        save_path = os.path.join(IMAGES_PATH, name)

        raw = rawpy.imread(from_dir)
        rgb = raw.postprocess()
        imageio.imsave(save_path, rgb)

        file_data = {'original_name': file_name}
        self.data[name] = file_data

    def _jpg2png(self, file_name, e):
        from_dir = os.path.join(RAW_IMAGES_PATH, file_name)
        name = str(e) + '.png'
        save_path = os.path.join(IMAGES_PATH, name)

        im = Image.open(from_dir)
        im.save(save_path)

        file_data = {'original_name': file_name}
        self.data[name] = file_data
        
    def _save_pickle(self):
        for name in self.data:
            self.data[name]['W'] = 0 # resolution width
            self.data[name]['H'] = 0 # resolution height
            self.data[name]['body_centerX'] = 0 # center possition of body
            self.data[name]['body_centerY'] = 0
            self.data[name]['square_lenght'] = 0 # lenght of body cut box
            self.data[name]['angle'] = False # value of rotated angle in degrees
            self.data[name]['add_black'] = 0 # list with added blact stripes if needed
            self.data[name]['body_is_centered'] = 0 # True if body was found
            self.data[name]['is_randomly_oriented'] = 0 # True if program had to randomly oriented image
            self.data[name]['class1'] = [] # labeled points class 1, left scales
            self.data[name]['class2'] = [] # labeled points class 2, right scales
            self.data[name]['class1_CNN'] = [] # found points class 1, left scales
            self.data[name]['class2_CNN'] = [] # found points class 2, right scales
            self.data[name]['diff'] = None # difference between lenghts of found and labeled
            self.data[name]['status'] = None # OK if program found number of scales in input range

        pickle_out = open(PICKLE_PATH, "wb")
        pickle.dump(self.data, pickle_out)
        pickle_out.close()

        print('done')


if __name__ == "__main__":
    obj = Prepare()
    obj.randomize()