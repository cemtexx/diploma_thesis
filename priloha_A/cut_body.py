import os
import pickle

from tqdm import tqdm
import numpy as np
import cv2

from configure import *

# set colors for bboxes
labels = open(LABELS_FILE).read().strip().split("\n")
np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(labels), 3),
                           dtype="uint8")


class CutBody():
    def __init__(self):
        # make dir for cuts images
        try:
            os.mkdir(BODIES_PATH)
        except FileExistsError:
            pass

        # make dir for labeled images
        try:
            os.mkdir(YOLO_LABELED_PATH)
        except FileExistsError:
            pass

        # make dataset list
        self.dataset = []
        for file_name in os.listdir(IMAGES_PATH):
            if file_name[-4:] == '.png':
                self.dataset.append(file_name)

        # sort
        self.dataset = [
            str(i)+".png" for i in sorted([int(num.split('.')[0]) for num in self.dataset])]

    def cutting(self):  # for all dataset
        print('start yolo evaluating and cutting bodies')
        self.all_cut_info = {}
        for file_name in tqdm(self.dataset, total=len(self.dataset)):
            self._yolo_recognizing(file_name)
            cut_info = self._orienting(file_name)
            self._save_data(file_name, cut_info)

        self._save_pickle()
        print('done')

    def cut_one(self, file_name):  # just for one image
        self._yolo_recognizing(file_name)
        cut_info = self._orienting(file_name)
        self._save_data(file_name, cut_info)

    def _yolo_recognizing(self, file_name):
        # credit for getting boxes: https://github.com/opencv/opencv/issues/19252
        self.input_file = os.path.join(IMAGES_PATH, file_name)

        net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

        self.image = cv2.imread(self.input_file)
        self.H = self.image.shape[0]
        self.W = self.image.shape[1]

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        self.boxes = []
        self.confidences = []
        self.classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions
                if confidence > CONFIDENCE_THRESHOLD:
                    box = detection[0:4] * \
                        np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    self.boxes.append(
                        [x, y, int(width), int(height), centerX, centerY])
                    self.confidences.append(float(confidence))
                    self.classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        self.idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, CONFIDENCE_THRESHOLD,
                                     CONFIDENCE_THRESHOLD)

    def _orienting(self, file_name):
        found_body = []  # [certainity, centrx, centry, width, height, square_lenght]
        found_head = []  # [certainity, centrx, centry, width, height]
        found_tail = []  # [certainity, centrx, centry, width, height]

        labeled_img = np.copy(self.image)

        # ensure at least one detection exists
        if len(self.idxs) > 0:
            # loop over the indexes we are keeping
            for i in self.idxs.flatten():
                # extract the bounding box coordinates
                (w, h) = (self.boxes[i][2], self.boxes[i][3])
                (x, y) = (self.boxes[i][0], self.boxes[i][1])

                (centerX, centerY) = (self.boxes[i][4], self.boxes[i][5])

                # pick the most confident box from each class
                # body
                if self.classIDs[i] == 0 and (len(found_body) == 0 or found_body[0] < self.confidences[i]):
                    found_body = [self.confidences[i],
                                  centerX, centerY, w, h, 0, False, True]
                    # +(max(found_body[3], found_body[4])*ENLARGE_BODY)
                    found_body[5] = max(found_body[3], found_body[4])

                # head
                elif self.classIDs[i] == 1 and (len(found_head) == 0 or found_head[0] < self.confidences[i]):
                    found_head = [self.confidences[i], centerX, centerY]

                # tail
                elif self.classIDs[i] == 2 and (len(found_tail) == 0 or found_tail[0] < self.confidences[i]):
                    found_tail = [self.confidences[i], centerX, centerY]

                color = [int(c) for c in COLORS[self.classIDs[i]]]

                cv2.rectangle(labeled_img, (x, y), (x + w, y + h), color, 8)
                text = "{}: {:.4f}".format(
                    labels[self.classIDs[i]], self.confidences[i])
                cv2.putText(labeled_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            2, color, 5)

        # save labeled image
        save_path = os.path.join(YOLO_LABELED_PATH, file_name)
        cv2.imwrite(save_path, labeled_img)

        if len(found_body) == 0:
            angle = 0
            found_body = [0, 0, 0, 0, 0, 0, True, False]
            black = [0, 0, 0, 0]
        elif len(found_body) != 0 and len(found_head) == 0 and len(found_tail) == 0:
            found_body[6] = True
            if found_body[3] >= found_body[4]:  # horizontally
                angle = 0
                black = self._cut_square(
                    self.image, found_body[1], found_body[2], found_body[5]*ENLARGE_BODY, angle, file_name)
            else:  # vertically
                angle = 90
                black = self._cut_square(
                    self.image, found_body[1], found_body[2], found_body[5]*ENLARGE_BODY, angle, file_name)
        elif len(found_body) != 0 and len(found_tail) != 0:
            if found_body[3] >= found_body[4]:  # horizontally
                if found_body[1] <= found_tail[1]:  # tail is on right
                    angle = 0
                    black = self._cut_square(
                        self.image, found_body[1], found_body[2], found_body[5]*ENLARGE_BODY, angle, file_name)
                else:  # tail is on left
                    angle = 180
                    black = self._cut_square(
                        self.image, found_body[1], found_body[2], found_body[5]*ENLARGE_BODY, angle, file_name)
            else:  # vertically
                if found_body[2] <= found_tail[2]:  # tail is on bottom
                    angle = 90
                    black = self._cut_square(
                        self.image, found_body[1], found_body[2], found_body[5]*ENLARGE_BODY, angle, file_name)
                else:  # tail is on top
                    angle = 270
                    black = self._cut_square(
                        self.image, found_body[1], found_body[2], found_body[5]*ENLARGE_BODY, angle, file_name)

        elif len(found_body) != 0 and len(found_tail) == 0 and len(found_head) != 0:
            if found_body[3] >= found_body[4]:  # horizontally
                if found_body[1] <= found_head[1]:  # head is on right
                    angle = 180
                    black = self._cut_square(
                        self.image, found_body[1], found_body[2], found_body[5]*ENLARGE_BODY, angle, file_name)
                else:  # head is on left
                    angle = 0
                    black = self._cut_square(
                        self.image, found_body[1], found_body[2], found_body[5]*ENLARGE_BODY, angle, file_name)
            else:  # vertically
                if found_body[2] <= found_head[2]:  # head is on bottom
                    angle = 270
                    black = self._cut_square(
                        self.image, found_body[1], found_body[2], found_body[5]*ENLARGE_BODY, angle, file_name)
                else:  # head is on top
                    angle = 90
                    black = self._cut_square(
                        self.image, found_body[1], found_body[2], found_body[5]*ENLARGE_BODY, angle, file_name)

        cut_info = {}
        cut_info['W'] = self.W
        cut_info['H'] = self.H
        cut_info['body_centerX'] = found_body[1]
        cut_info['body_centerY'] = found_body[2]
        cut_info['square_lenght'] = found_body[5]
        cut_info['angle'] = angle
        cut_info['add_black'] = black
        cut_info['body_is_centered'] = found_body[7]
        cut_info['is_randomly_oriented'] = found_body[6]

        return cut_info

    def _cut_square(self, image, centerX, centerY, lenght, inverse, name):
        add_black = [0, 0, 0, 0]  # [top, bottom, left, right]

        fromY = int(centerY-(lenght/2))
        toY = int(centerY+(lenght/2))
        fromX = int(centerX-(lenght/2))
        toX = int(centerX+(lenght/2))

        if fromY < 0:  # body is over top boarder
            add_black[0] = abs(fromY)
        if fromX < 0:  # body is over left boarder
            add_black[2] = abs(fromX)
        if toX > self.W:  # body is over right boarder
            add_black[3] = abs(toX-self.W)
        if toY > self.H:  # body is over bottom boarder
            add_black[1] = abs(toY-self.H)

        # add black stripes if need
        if sum(add_black) != 0:
            bigger = np.zeros(
                (self.H + add_black[0] + add_black[1], self.W + add_black[2] + add_black[3], 3), np.uint8)
            bigger[add_black[0]:add_black[0] + image.shape[0],
                   add_black[2]:add_black[2]+image.shape[1]] = image
            cut = bigger[fromY+add_black[0]:toY+add_black[0],
                         fromX+add_black[2]:toX+add_black[2]]
        else:
            cut = image[fromY:toY, fromX:toX]

        # rotate with image
        if inverse == 90:
            cut = cv2.rotate(cut, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif inverse == 270:
            cut = cv2.rotate(cut, cv2.ROTATE_90_CLOCKWISE)
        elif inverse == 180:
            cut = cv2.rotate(cut, cv2.ROTATE_180)

        # resized image to CUT_SIZE
        resized = cv2.resize(cut, (CUT_SIZE, CUT_SIZE),
                             interpolation=cv2.INTER_AREA)
        save_path = os.path.join(BODIES_PATH, name)
        cv2.imwrite(save_path, resized)
        return add_black

    def _save_data(self, file_name, cut_info):
        self.all_cut_info[file_name] = cut_info

    def _save_pickle(self):
        pickle_in = open(PICKLE_PATH, 'rb')
        images_data = pickle.load(pickle_in)
        pickle_in.close()
        for image in images_data:
            for key_cut_info in self.all_cut_info[image]:
                images_data[image][key_cut_info] = self.all_cut_info[image][key_cut_info]

        pickle_out = open(PICKLE_PATH, "wb")
        pickle.dump(images_data, pickle_out)
        pickle_out.close()


if __name__ == "__main__":
    obj = CutBody()
    obj.cutting()
