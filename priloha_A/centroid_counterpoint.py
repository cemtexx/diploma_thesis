import os

# turn off tensorflow warning informations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2

from configure import *


def no_neighbor(point, points, minimum_distance):
    if len(points) == 0:
        return True

    for loop_point in points:
        distance = np.sqrt((point[0] - loop_point[0])
                           ** 2 + (point[1] - loop_point[1])**2)
        if distance < minimum_distance:
            return False

    return True


def counterpoint(raw_image, local_filter, minimum_distance, tm, centers_to=None):
    image = raw_image*(255/raw_image.max())
    image = image.astype(int)

    if (local_filter % 2) == 0:
        local_filter += 1

    H = image.shape[0]
    W = image.shape[1]

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

    # DEVIDE INTO CLASS1 (LEFT SCALES) AND CLASS2 (RIGHT SCALES)
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

    # DRAW RESULTS INTO A IMAGES
    final_image = np.append(image, image, axis=2)
    final_image = np.append(final_image, image, axis=2)

    for center in class1:
        final_image = cv2.circle(
            final_image, (center[1], center[0]), 0, (0, 0, 255), 3)
    for center in class2:
        final_image = cv2.circle(
            final_image, (center[1], center[0]), 0, (255, 0, 0), 3)

    centers_original = centers_to.copy()
    if centers_to is not None:
        for center in class1:  # cervena je dole
            centers_original = cv2.circle(
                centers_original, (center[1], center[0]), 0, (0, 0, 255), 3)
        for center in class2:
            centers_original = cv2.circle(
                centers_original, (center[1], center[0]), 0, (255, 0, 0), 3)

    return [[class1, class2], final_image, centers_original]
