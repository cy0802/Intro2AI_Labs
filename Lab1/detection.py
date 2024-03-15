import os
from unittest import result
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt
import classifier


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:A
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    """
      First, I extract the txt file at the `dataPath` and get the information of images and the position of face area.
      For each image, I read the image, crop the face area, and resize the face area to 19 x 19.
      For each face area, I pass it to the `clf.classify` function to get the result.
      If the result is True, I draw a green box on the image; otherwise, draw a red box on the image.
    """
    with open(dataPath, "r") as file:
      raw_data = file.read()
    lines = raw_data.split("\n")
    line_idx = 0
    while line_idx < len(lines) - 1:
      img_path = lines[line_idx].split(" ")[0]
      num_box = lines[line_idx].split(" ")[1]
      img = cv2.imread("data/detect/" + img_path)
      for i in range(int(num_box)):
        line_idx += 1
        box = lines[line_idx].split(" ")
        # print(box)
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (19, 19))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        if clf.classify(face):
          cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        else:
          cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 5)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      plt.imshow(img)
      plt.show()
      line_idx += 1
    # End your code (Part 4)
