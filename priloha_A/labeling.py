import os
import pickle

import cv2
import numpy as np

from configure import *

# constant for zooming lizard
# does not affect final result
ZOOM = 2


class ClickGradients():
    def __init__(self) -> None:
        try:
            # case of already exists some data
            pickle_in = open(PICKLE_PATH, 'rb')
            self.gradients = pickle.load(pickle_in)
            pickle_in.close()

        except FileNotFoundError:
            print('Žádná label data')
            self.gradients = {}

        # make a dataset
        self.dataset = []
        for file_name in os.listdir(BODIES_PATH):
            if file_name[-4:] == '.png':
                self.dataset.append(file_name)

        # sort
        self.dataset = [
            str(i)+".png" for i in sorted([int(num.split('.')[0]) for num in self.dataset])]

        # just first value, user can change
        self.gradient_radius = 5

    def label_images(self):
        self.file_ord = 0
        while self.file_ord < len(self.dataset):
            file_name = self.dataset[self.file_ord]

            if 'class1' in self.gradients[file_name]:
                class1 = self.gradients[file_name]['class1']
            else:
                class1 = []  # left scales

            if 'class2' in self.gradients[file_name]:
                class2 = self.gradients[file_name]['class2']
            else:
                class2 = []  # right scales

            file_path = os.path.join(BODIES_PATH, file_name)
            raw_image = cv2.imread(file_path)
            (H, W) = raw_image.shape[:2]
            # raw_image = cv2.resize(raw_image, (H*2, W*2), interpolation = cv2.INTER_AREA)

            (H, W) = raw_image.shape[:2]

            raw_and_black = np.zeros((H, 2*W, 3), np.uint8)
            raw_and_black[0:H, 0:W] = raw_image
            # raw_and_black = cv2.rectangle(raw_and_black, (W, 0), (W*2-1, H-1), (255, 255, 255), 1)

            self.mouse_x = 10
            self.mouse_y = 10

            def class1_mark(event, x, y, flags, param):
                if event == cv2.EVENT_MOUSEMOVE:
                    self.mouse_x = int(x/ZOOM)
                    self.mouse_y = int(y/ZOOM)

                if event == cv2.EVENT_LBUTTONDOWN and x/ZOOM <= W:
                    class1.append(
                        (int(x/ZOOM), int(y/ZOOM), self.gradient_radius))
                if event == cv2.EVENT_RBUTTONDOWN:
                    class1.pop()
                if event == cv2.EVENT_MOUSEWHEEL:
                    if flags > 0:
                        self.gradient_radius -= 1
                    else:
                        self.gradient_radius += 1

                    if self.gradient_radius > 7:
                        self.gradient_radius = 7
                    elif self.gradient_radius < 3:
                        self.gradient_radius = 3

            def class2_mark(event, x, y, flags, param):
                if event == cv2.EVENT_MOUSEMOVE:
                    self.mouse_x = int(x/ZOOM)
                    self.mouse_y = int(y/ZOOM)

                if event == cv2.EVENT_LBUTTONDOWN and x/ZOOM <= W:
                    class2.append(
                        (int(x/ZOOM), int(y/ZOOM), self.gradient_radius))
                if event == cv2.EVENT_RBUTTONDOWN:
                    class2.pop()
                if event == cv2.EVENT_MOUSEWHEEL:
                    if flags > 0:
                        self.gradient_radius -= 1
                    else:
                        self.gradient_radius += 1

                    if self.gradient_radius > 7:
                        self.gradient_radius = 7
                    elif self.gradient_radius < 3:
                        self.gradient_radius = 3

            cv2.namedWindow('draw')

            cv2.setMouseCallback('draw', class1_mark)

            jump = False

            while True:
                image = np.copy(raw_and_black)
                image = cv2.rectangle(
                    image, (5, 5), (30, 45), (255, 255, 255), -1)
                cv2.circle(image, (17, 17),
                           self.gradient_radius, (0, 0, 255), 1)
                cv2.circle(image, (self.mouse_x, self.mouse_y),
                           self.gradient_radius, (0, 0, 255), 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, str(self.gradient_radius),
                            (15, 34), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    image, file_name[:-3], (8, 44), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

                for point in class1:
                    cv2.circle(image, (point[0], point[1]),
                               point[2], (0, 0, 255), 1)
                    image = self._mark_gradient(
                        point[1], point[0] + W, image, point[2])
                for point in class2:
                    cv2.circle(image, (point[0], point[1]),
                               point[2], (255, 0, 0), 1)
                    image = self._mark_gradient(
                        point[1], point[0] + W, image, point[2])

                bigger_image = cv2.resize(
                    image, (image.shape[1]*ZOOM, image.shape[0]*ZOOM), interpolation=cv2.INTER_AREA)
                cv2.imshow('draw', bigger_image)

                if cv2.waitKey(1) & 0xFF == ord('e'):
                    break

                if cv2.waitKey(1) & 0xFF == ord('r'):
                    self.file_ord += 9
                    jump = True
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.file_ord -= 2
                    jump = True
                    break

                del image
                del bigger_image

            cv2.setMouseCallback('draw', class2_mark)
            if not jump:
                while True:
                    image = np.copy(raw_and_black)
                    image = cv2.rectangle(
                        image, (5, 5), (30, 45), (255, 255, 255), -1)
                    cv2.circle(image, (17, 17),
                               self.gradient_radius, (255, 0, 0), 1)
                    cv2.circle(image, (self.mouse_x, self.mouse_y),
                               self.gradient_radius, (255, 0, 0), 1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, str(self.gradient_radius),
                                (15, 34), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(
                        image, file_name[:-3], (8, 44), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA)

                    for point in class1:
                        cv2.circle(
                            image, (point[0], point[1]), point[2], (0, 0, 255), 1)
                        image = self._mark_gradient(
                            point[1], point[0] + W, image, point[2])
                    for point in class2:
                        cv2.circle(
                            image, (point[0], point[1]), point[2], (255, 0, 0), 1)
                        image = self._mark_gradient(
                            point[1], point[0] + W, image, point[2])

                    bigger_image = cv2.resize(
                        image, (image.shape[1]*ZOOM, image.shape[0]*ZOOM), interpolation=cv2.INTER_AREA)
                    cv2.imshow('draw', bigger_image)

                    if cv2.waitKey(1) & 0xFF == ord('e'):
                        break

                    if cv2.waitKey(1) & 0xFF == ord('r'):
                        self.file_ord += 9
                        break

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.file_ord -= 2
                        break

                    del image
                    del bigger_image

            self.file_ord += 1

            if self.file_ord < 0:
                self.file_ord = 0

            gradients_local = [class1, class2]

            self._save_pickle(gradients_local, file_name)

            cv2.imwrite('example.png', bigger_image)

        print('end of the program')

    def _save_pickle(self, data, file_name):
        self.gradients[file_name]['class1'] = data[0]
        self.gradients[file_name]['class2'] = data[1]

        pickle_out = open(PICKLE_PATH, "wb")
        pickle.dump(self.gradients, pickle_out)
        pickle_out.close()

        print(file_name, 'saved')

    def _mark_gradient(self, x, y, image, radius):
        for i in range(1, radius*2+1):
            for j in range(1, radius*2+1):
                distance = np.sqrt((i-radius)**2 + (j-radius)**2)
                if distance < radius:
                    # linearni interpolace
                    # gray = int(-(255/radius)*distance + 255)

                    # nelinearni interpolace
                    nonlinear_parameter = 3  # the higher the more nonlinear
                    gray = -255*(distance/radius)**(1 /
                                                    nonlinear_parameter) + 255

                    try:
                        image[i+x-radius, j+y-radius] = (gray, gray, gray)
                    except IndexError:
                        pass
        return image


if __name__ == "__main__":
    obj = ClickGradients()
    obj.label_images()
