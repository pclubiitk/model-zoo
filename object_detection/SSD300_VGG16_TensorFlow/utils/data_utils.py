import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np

def read_sample(image_path, label_path):

    image_path = image_path.strip("\n")
    label_path = label_path.strip("\n")
    assert os.path.exists(image_path), "Image file does not exist."
    assert os.path.exists(label_path), "Label file does not exist."

    image = cv2.imread(image_path)  # read image in bgr format
    bboxes, classes = [], []
    xml_root = ET.parse(label_path).getroot()
    objects = xml_root.findall("object")
    for i, obj in enumerate(objects):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        # the reason why we use float() is because some value in bndbox are float
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        bboxes.append([xmin, ymin, xmax, ymax])
        classes.append(name)
    return np.array(image, dtype=np.float), np.array(bboxes, dtype=np.float), classes

def generate_samples_from_split(split_file, images_dir,  xml_dir):

    assert os.path.isfile(split_file), "split_file does not exists."
    assert os.path.isdir(images_dir), "images_dir is not a directory."
    assert os.path.isdir(xml_dir), "xml_dir is not a directory."

    samples = []

    with open(split_file, "r") as split_file:
        lines = split_file.readlines()
        for line in lines:
            image_file = os.path.join(images_dir, line.strip("\n") + ".jpg")
            xml_file = os.path.join(xml_dir, line.strip("\n") + ".xml")
            sample = f"{image_file} {xml_file}"
            samples.append(sample)

    return samples
