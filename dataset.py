import os
import cv2
import glob
import numpy as np

def load_data_small():
    """
        This function loads images form the path: 'data/data_small' and return the training
        and testing dataset. The dataset is a list of tuples where the first element is the 
        numpy array of shape (m, n) representing the image the second element is its 
        classification (1 or 0).

        Parameters:
            None

        Returns:
            dataset: The first and second element represents the training and testing dataset respectively
    """

    # Begin your code (Part 1-1)
    print("load_data_small()")
    train_dataset, test_dataset = [], []
    path = [("data/data_small/test/face/", 1), ("data/data_small/test/non-face/", 0), 
                ("data/data_small/train/face/", 1), ("data/data_small/train/non-face/", 0)]
    for i in range(len(path)):
        files = os.listdir(path[i][0])
        for file in files:
            img = cv2.imread(path[i][0] + file, cv2.IMREAD_GRAYSCALE)
            if i < 2:
                test_dataset.append((img, path[i][1]))
            else:
                train_dataset.append((img, path[i][1]))
    dataset = (train_dataset, test_dataset)
    # End your code (Part 1-1)
    
    return dataset


def load_data_FDDB(data_idx="01"):
    """
        This function generates the training and testing dataset form the path: 'data/data_small'.
        The dataset is a list of tuples where the first element is the numpy array of shape (m, n)
        representing the image the second element is its classification (1 or 0).
        
        In the following, there are 4 main steps:
        1. Read the .txt file
        2. Crop the faces using the ground truth label in the .txt file
        3. Random crop the non-faces region
        4. Split the dataset into training dataset and testing dataset
        
        Parameters:
            data_idx: the data index string of the .txt file

        Returns:
            train_dataset: the training dataset
            test_dataset: the testing dataset
    """

    with open("data/data_FDDB/FDDB-folds/FDDB-fold-{}-ellipseList.txt".format(data_idx)) as file:
        line_list = [line.rstrip() for line in file]

    # Set random seed for reproducing same image croping results
    np.random.seed(0)

    face_dataset, nonface_dataset = [], []
    line_idx = 0

    # Iterate through the .txt file
    # The detail .txt file structure can be seen in the README at https://vis-www.cs.umass.edu/fddb/
    while line_idx < len(line_list):
        # read a image
        img_gray = cv2.imread(os.path.join("data/data_FDDB", line_list[line_idx] + ".jpg"), cv2.IMREAD_GRAYSCALE)
        # print(f"image: {line_list[line_idx]}, shape: {img_gray.shape}")
        num_faces = int(line_list[line_idx + 1])

        # Crop face region using the ground truth label
        face_box_list = []
        for i in range(num_faces):
            # Here, each face is denoted by:
            # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
            coord = [int(float(j)) for j in line_list[line_idx + 2 + i].split()]
            x, y = coord[3] - coord[1], coord[4] - coord[0]            
            w, h = 2 * coord[1], 2 * coord[0]

            left_top = (max(x, 0), max(y, 0))
            right_bottom = (min(x + w, img_gray.shape[1]), min(y + h, img_gray.shape[0]))
            face_box_list.append([left_top, right_bottom])
            # cv2.rectangle(img_gray, left_top, right_bottom, (0, 255, 0), 2)

            img_crop = img_gray[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]].copy()
            face_dataset.append((cv2.resize(img_crop, (19, 19)), 1))

        line_idx += num_faces + 2

        # Random crop N non-face region
        # Here we set N equal to the number of faces to generate a balanced dataset
        # Note that we have alreadly save the bounding box of faces into `face_box_list`, you can utilize it for non-face region cropping
        # print(face_box_list)
        for i in range(num_faces):
            # Begin your code (Part 1-2)
            coord = []
            while True:
                coord1 = (np.random.randint(0, img_gray.shape[1]-40), np.random.randint(0, img_gray.shape[0]-40))
                coord2 = (coord1[0] + 19, coord1[1] + 19)
                overlap_flag = False
                for box in face_box_list:
                    if overlap(box, (coord1, coord2)):
                        overlap_flag = True
                        break
                if not overlap_flag:
                    coord = [coord1, coord2]
                    break
            # print(f"non-face: {coord}")
            img_crop = img_gray[coord[0][1]:coord[1][1], coord[0][0]:coord[1][0]].copy()
            # print(img_crop.shape)
            nonface_dataset.append((img_crop, 0))
            # End your code (Part 1-2)

        # cv2.imshow("windows", img_gray)
        # cv2.waitKey(0)

    # train test split
    print("face_dataset: ", len(face_dataset), "\nnonface_dataset: ", len(nonface_dataset))
    num_face_data, num_nonface_data = len(face_dataset), len(nonface_dataset)
    SPLIT_RATIO = 0.7

    train_dataset = face_dataset[:int(SPLIT_RATIO * num_face_data)] + nonface_dataset[:int(SPLIT_RATIO * num_nonface_data)]
    test_dataset = face_dataset[int(SPLIT_RATIO * num_face_data):] + nonface_dataset[int(SPLIT_RATIO * num_nonface_data):]

    return train_dataset, test_dataset

def overlap(rect1, rect2):
    # rect = [(left_top_x, left_top_y), (right_bottom_x, right_bottom_y)]
    corner = [(rect1[0][0], rect1[0][1]), (rect1[0][0], rect1[1][0]), 
                (rect1[1][0], rect1[0][1]), (rect1[1][0], rect1[1][1])]
    for point in corner:
        if rect2[0][0] <= point[0] and point[0] <= rect2[1][0] and \
            rect2[0][1] <= point[1] and point[1] <= rect2[1][1]:
            return True
    return False

def create_dataset(data_type):
    if data_type == "small":
        return load_data_small()
    else:
        return load_data_FDDB()

if __name__ == "__main__":
    train, test = load_data_FDDB()
    print(len(train), len(test))