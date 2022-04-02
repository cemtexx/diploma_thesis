import os

import yaml
import cv2
import numpy as np
from psd_tools import PSDImage
import rawpy
import imageio
from skimage.io import imread

# load configuration file
config_path = 'config.yaml'
with open(config_path, 'rb') as f:
    config = yaml.load(f, Loader=yaml.Loader)


class ScalesCounter():
    def __init__(self) -> None:
        self.error_info = 'OK'

    def yolo_recognizing(self, image_path, net):
        if image_path[-4:] == '.jpg' or image_path[-4:] == '.JPG' or image_path[-4:] == '.png' or image_path[-4:] == '.PNG':
            #raw_image = cv2.imread(image_path)
            raw_image = imread(image_path)[:, :, :3]

        elif image_path[-4:] == '.psd' or image_path[-4:] == '.PSD':
            psd = PSDImage.open(image_path)
            psd.composite().save(config['TEMPORARY_IMAGE'])
            #raw_image = cv2.imread(TEMPORARY_IMAGE)
            raw_image = imread(config['TEMPORARY_IMAGE'])[:, :, :3]
            os.remove(config['TEMPORARY_IMAGE'])

        elif image_path[-4:] == '.CR2':
            raw = rawpy.imread(image_path)
            rgb = raw.postprocess()
            imageio.imsave(config['TEMPORARY_IMAGE'], rgb)
            #raw_image = cv2.imread(TEMPORARY_IMAGE)
            raw_image = imread(config['TEMPORARY_IMAGE'])[:, :, :3]
            os.remove(config['TEMPORARY_IMAGE'])

        self.H = raw_image.shape[0]
        self.W = raw_image.shape[1]

        # credit for getting boxes: https://github.com/opencv/opencv/issues/19252
        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(raw_image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions
                if confidence > config['CONFIDENCE_THRESHOLD']:
                    box = detection[0:4] * \
                        np.array([self.W, self.H, self.W, self.H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append(
                        [x, y, int(width), int(height), centerX, centerY])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, config['CONFIDENCE_THRESHOLD'],
                                config['CONFIDENCE_THRESHOLD'])

        found_body = []  # [certainity, centrx, centry, width, height, square_lenght]
        found_head = []  # [certainity, centrx, centry, width, height]
        found_tail = []  # [certainity, centrx, centry, width, height]

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (w, h) = (boxes[i][2], boxes[i][3])
                (x, y) = (boxes[i][0], boxes[i][1])

                (centerX, centerY) = (boxes[i][4], boxes[i][5])

                # pick the most confident box from each class
                # body
                if classIDs[i] == 0 and (len(found_body) == 0 or found_body[0] < confidences[i]):
                    found_body = [confidences[i], centerX, centerY, w, h, 0]
                    found_body[5] = max(found_body[3], found_body[4])

                # head
                elif classIDs[i] == 1 and (len(found_head) == 0 or found_head[0] < confidences[i]):
                    found_head = [confidences[i], centerX, centerY]

                # tail
                elif classIDs[i] == 2 and (len(found_tail) == 0 or found_tail[0] < confidences[i]):
                    found_tail = [confidences[i], centerX, centerY]

        # case of none found body
        if len(found_body) == 0:
            self.error_info = 'body was not found'
            return None

        elif len(found_body) != 0 and len(found_head) == 0 and len(found_tail) == 0:

            if found_body[3] >= found_body[4]:  # horizontally
                angle = 0
                cut_image = self._cut_square(
                    raw_image, found_body[1], found_body[2], found_body[5]*config['ENLARGE_BODY'], angle)
            else:  # vertically
                angle = 90
                cut_image = self._cut_square(
                    raw_image, found_body[1], found_body[2], found_body[5]*config['ENLARGE_BODY'], angle)
            self.error_info = 'body is randomly oriented'

        elif len(found_body) != 0 and len(found_tail) != 0:
            if found_body[3] >= found_body[4]:  # horizontally
                if found_body[1] <= found_tail[1]:  # tail is on right
                    angle = 0
                    cut_image = self._cut_square(
                        raw_image, found_body[1], found_body[2], found_body[5]*config['ENLARGE_BODY'], angle)
                else:  # tail is on left
                    angle = 180
                    cut_image = self._cut_square(
                        raw_image, found_body[1], found_body[2], found_body[5]*config['ENLARGE_BODY'], angle)
            else:  # vertically
                if found_body[2] <= found_tail[2]:  # tail is on bottom
                    angle = 90
                    cut_image = self._cut_square(
                        raw_image, found_body[1], found_body[2], found_body[5]*config['ENLARGE_BODY'], angle)
                else:  # tail is on top
                    angle = 270
                    cut_image = self._cut_square(
                        raw_image, found_body[1], found_body[2], found_body[5]*config['ENLARGE_BODY'], angle)

        elif len(found_body) != 0 and len(found_tail) == 0 and len(found_head) != 0:
            if found_body[3] >= found_body[4]:  # horizontally
                if found_body[1] <= found_head[1]:  # head is on right
                    angle = 180
                    cut_image = self._cut_square(
                        raw_image, found_body[1], found_body[2], found_body[5]*config['ENLARGE_BODY'], angle)
                else:  # head is on left
                    angle = 0
                    cut_image = self._cut_square(
                        raw_image, found_body[1], found_body[2], found_body[5]*config['ENLARGE_BODY'], angle)
            else:  # vertically
                if found_body[2] <= found_head[2]:  # head is on bottom
                    angle = 270
                    cut_image = self._cut_square(
                        raw_image, found_body[1], found_body[2], found_body[5]*config['ENLARGE_BODY'], angle)
                else:  # head is on top
                    angle = 90
                    cut_image = self._cut_square(
                        raw_image, found_body[1], found_body[2], found_body[5]*config['ENLARGE_BODY'], angle)

        return cut_image

    def _cut_square(self, image, centerX, centerY, lenght, inverse):
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
        resized = cv2.resize(
            cut, (config['CUT_SIZE'], config['CUT_SIZE']), interpolation=cv2.INTER_AREA)
        return resized

    def u_net(self, cut_image, models):
        # the first try
        preds_model = models[0].predict(
            np.expand_dims(cut_image/255, axis=0))[0]
        results = self._counterpoint(preds_model, cut_image)[:]
        # the second try if the first was not correct
        if results[0] > config['MAXIMUM_SCALES'] or results[0] < config['MINIMUM_SCALES'] or results[1] > config['MAXIMUM_SCALES'] or results[1] < config['MINIMUM_SCALES']:
            second_preds_model = models[1].predict(
                np.expand_dims(cut_image/255, axis=0))[0]
            results = self._counterpoint(second_preds_model, cut_image)[:]

            # the third try if the second was not correct
            if results[0] > config['MAXIMUM_SCALES'] or results[0] < config['MINIMUM_SCALES'] or results[1] > config['MAXIMUM_SCALES'] or results[1] < config['MINIMUM_SCALES']:
                third_preds_model = models[2].predict(
                    np.expand_dims(cut_image/255, axis=0))[0]
                results = self._counterpoint(third_preds_model, cut_image)[:]

                # all attempts were not correct
                if results[0] > config['MAXIMUM_SCALES'] or results[0] < config['MINIMUM_SCALES'] or results[1] > config['MAXIMUM_SCALES'] or results[1] < config['MINIMUM_SCALES']:
                    self.error_info = 'program did not count correctly'

        return results

    def _counterpoint(self, preds_model, raw_image):
        # some preprocess operations
        image = preds_model*(255/preds_model.max())
        image = image.astype(int)
        H = image.shape[0]
        W = image.shape[1]

        # define and check local filter size
        local_filter = config['LOCAL_FILTER']
        if (local_filter % 2) == 0:
            local_filter += 1

        # MAX FILTER
        add_black = int(local_filter/2)
        padding = np.pad(np.squeeze(image), pad_width=add_black,
                         mode='constant', constant_values=0)
        padding = np.expand_dims(padding, axis=2)
        max_filter = np.zeros((H, W, 1), np.uint8)
        for x in range(0, H):
            for y in range(0, W):
                im_filter = padding[x:x+local_filter, y:y+local_filter]
                max_filter[x][y] = im_filter.max()

        B1 = np.zeros((H, W, 1), np.uint8)
        for x in range(0, H):
            for y in range(0, W):
                if max_filter[x][y] == image[x][y]:
                    B1[x][y] = 255
                else:
                    B1[x][y] = 0

        # MASK OMEGA
        omega = np.zeros((H, W, 1), np.uint8)
        for x in range(0, H):
            for y in range(0, W):
                if image[x][y] == 0:
                    omega[x][y] = 255
                else:
                    omega[x][y] = 0

        # EROSION
        kernel = np.ones((local_filter, local_filter), np.uint8)
        erosion = cv2.erode(omega, kernel, iterations = 1)

        # XOR OF EROSION MASK AND MASK1
        B_omega = np.zeros((H, W, 1), np.uint8)
        for x in range(0, H):
            for y in range(0, W):
                if B1[x][y] == erosion[x][y]:
                    B_omega[x][y] = 0
                else:
                    B_omega[x][y] = 255


        # MASK1
        centroid_map = np.zeros((H, W, 1), np.uint8)
        centers = []
        for x in range(0, H):
            for y in range(0, W):
                if B_omega[x][y] == 255 and image[x][y] >= int(255*tm):
                    centroid_map[x][y] = 255
                    if no_neighbor((x, y), centers, MINIMUM_DISTANCE):
                        centers.append((x, y))
                else:
                    centroid_map[x][y] = 0

        image = image/255

        # devide into class1 (left scales) and class2 (right scales)
        group1 = []
        group2 = []

        most_left = (W, H)
        for point in centers:
            if point[1] < most_left[1]:
                most_left = point

        measured_point = most_left
        distances = []
        nearest_point = (0, 0)

        centers_to_devide = centers[:]
        centers_to_devide.remove(measured_point)
        group1.append(measured_point)

        # filter close points
        while True:
            shortest_dist = H
            for point in centers_to_devide:
                distance = np.sqrt(
                    (point[0] - measured_point[0])**2 + (point[1] - measured_point[1])**2)
                if distance < shortest_dist:
                    nearest_point = point
                    shortest_dist = distance

            measured_point = nearest_point

            if len(distances) == 0:
                pass
            elif shortest_dist > 2*(sum(distances)/len(distances)):
                group2 = centers_to_devide[:]
                break

            distances.append(shortest_dist)
            centers_to_devide.remove(measured_point)
            group1.append(measured_point)

        # decide which group is upper -> class2 (right scales) and which is more down -> class1 (left scales)
        sum_y = 0
        for point in group1:
            sum_y += point[0]
        group1_average_y = sum_y

        sum_y = 0
        for point in group2:
            sum_y += point[0]
        group2_average_y = sum_y

        if group1_average_y > group2_average_y:
            class1 = group1[:]
            class2 = group2[:]
        else:
            class1 = group2[:]
            class2 = group1[:]

        # draw results into images if requiered
        if config['IMAGE_OUTPUT']:
            final_image = raw_image.copy()
            for center in class1:  # blue is down as class1 left scales
                final_image = cv2.circle(
                    final_image, (center[1], center[0]), 0, (0, 0, 255), 3)
            for center in class2:  # red is up as class2 right scales
                final_image = cv2.circle(
                    final_image, (center[1], center[0]), 0, (255, 0, 0), 3)

            return [len(class1), len(class2), final_image]
        else:
            return [len(class1), len(class2)]

    def _no_neighbor(self, point, points):
        # return True if there is any close point
        if len(points) == 0:
            return True

        for loop_point in points:
            distance = np.sqrt((point[0] - loop_point[0])
                               ** 2 + (point[1] - loop_point[1])**2)
            if distance < config['MINIMUM_DISTANCE']:
                return False

        return True