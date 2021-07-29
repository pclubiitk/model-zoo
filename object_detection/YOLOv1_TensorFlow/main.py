import numpy as np
import argparse
import json
import xml.etree.ElementTree as ET
import os
from utils import (
    My_Custom_Generator,
    yolo_loss,
    CustomLearningRateScheduler,
    lr_schedule,
)
from model import yolo_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint


sets = [("2007", "train"), ("2007", "val"), ("2007", "test")]

classes_num = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}


# ***IMPORTANT : PASS YOUR train and test VOCdevkit paths here correctly!***
train_vocdevkit = "D:/work/final-yolo/VOCdevkit"
test_vocdevkit = "D:/work/final-yolo/testVOCdevkit"


def main():
    print("Inside the main function of main.py")

    config_file = open("config.json", "r")
    json_object = json.load(config_file)
    if (
        json_object["train_val_VOCdevkit_path"] == "None"
        or json_object["test_VOCdevkit_path"] == "None"
    ):
        print(
            "Please set the test and train dataset path correctly in the config.json file first."
        )
        return

    global train_vocdevkit
    global test_vocdevkit
    train_vocdevkit = str(json_object["train_val_VOCdevkit_path"])
    test_vocdevkit = str(json_object["test_VOCdevkit_path"])

    print(
        "Train VOCdevkit path set to {}\n Test VOCdevkit path set to {}".format(
            train_vocdevkit, test_vocdevkit
        )
    )

    parser = argparse.ArgumentParser(description="Build Annotations.")
    parser.add_argument("dir", default="..", help="Annotations.")

    for year, image_set in sets:
        print(train_vocdevkit)
        print(year, image_set)
        if image_set == "train":
            path_to_vocdevkit = train_vocdevkit
        elif image_set == "test":
            path_to_vocdevkit = test_vocdevkit
            print("path changed to test :{}".format(path_to_vocdevkit))
        with open(
            os.path.join(
                "{}/VOC{}/ImageSets/Main/{}.txt".format(
                    path_to_vocdevkit, year, image_set
                )
            ),
            "r",
        ) as f:
            image_ids = f.read().strip().split()
        with open(
            os.path.join("{}/{}_{}.txt".format(path_to_vocdevkit, year, image_set)), "w"
        ) as f:
            for image_id in image_ids:
                f.write(
                    "{}/VOC{}/JPEGImages/{}.jpg".format(
                        path_to_vocdevkit, year, image_id
                    )
                )
                convert_annotation(year, image_id, f, image_set)
                f.write("\n")

    # Preparing the input and output arrays
    print("Preparing inputs/outputs")
    train_datasets = []
    val_datasets = []
    test_datasets = []

    with open(os.path.join(train_vocdevkit, "2007_train.txt"), "r") as f:
        train_datasets = train_datasets + f.readlines()
    with open(os.path.join(train_vocdevkit, "2007_val.txt"), "r") as f:
        val_datasets = val_datasets + f.readlines()
    with open(os.path.join(test_vocdevkit, "2007_test.txt"), "r") as f:
        test_datasets = test_datasets + f.readlines()

    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    X_test = []
    Y_test = []

    for item in train_datasets:
        item = item.replace("\n", "").split(" ")
        X_train.append(item[0])
        arr = []
        for i in range(1, len(item)):
            arr.append(item[i])
        Y_train.append(arr)

    for item in val_datasets:
        item = item.replace("\n", "").split(" ")
        X_val.append(item[0])
        arr = []
        for i in range(1, len(item)):
            arr.append(item[i])
        Y_val.append(arr)

    for item in test_datasets:
        item = item.replace("\n", "").split(" ")
        X_test.append(item[0])
        arr = []
        for i in range(1, len(item)):
            arr.append(item[i])
        Y_test.append(arr)

    # Now let us create an instance of our custom generator for training and validation sets
    print("Genrating instance")
    batch_size = 4
    test_batch_size = 4

    # print("Y_train: {}".format(Y_train))

    my_training_batch_generator = My_Custom_Generator(X_train, Y_train, batch_size)

    my_validation_batch_generator = My_Custom_Generator(X_val, Y_val, batch_size)

    my_test_batch_generator = My_Custom_Generator(X_test, Y_test, test_batch_size)

    # return
    x_train, y_train = my_training_batch_generator.__getitem__(0)
    x_val, y_val = my_validation_batch_generator.__getitem__(0)
    x_test, y_test = my_test_batch_generator.__getitem__(0)

    print("Xtrain shape :{}".format(x_train.shape))
    print("Ytrain shape :{}".format(y_train.shape))
    print("Xval shape :{}".format(x_val.shape))
    print("Yval shape :{}".format(y_val.shape))
    print("Xtest shape :{}".format(x_test.shape))
    print("Ytest shape :{}".format(y_test.shape))

    # Getting the model ready
    print("Model making")
    model = yolo_model()
    mcp_save = ModelCheckpoint(
        "weight.hdf5", save_best_only=True, monitor="val_loss", mode="min"
    )
    model.compile(loss=yolo_loss, optimizer="adam")
    # Training the model
    model.fit(
        x=my_training_batch_generator,
        steps_per_epoch=int(len(X_train) // batch_size),
        epochs=135,
        verbose=1,
        workers=4,
        validation_data=my_validation_batch_generator,
        validation_steps=int(len(X_val) // batch_size),
        callbacks=[CustomLearningRateScheduler(lr_schedule), mcp_save],
    )
    # Training end
    y_pred = model.predict(x_test)
    np.save("results", y_pred)
    print("We are done training ! Results are stored in the file results.npy")


def convert_annotation(year, image_id, f, image_set):
    xx = train_vocdevkit
    if image_set == "train":
        xx = train_vocdevkit
    elif image_set == "test":
        xx = test_vocdevkit
        # print("path changed to test :{}".format(xx))
    in_file = os.path.join("%s/VOC%s/Annotations/%s.xml" % (xx, year, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        classes = list(classes_num.keys())
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        b = (
            int(xmlbox.find("xmin").text),
            int(xmlbox.find("ymin").text),
            int(xmlbox.find("xmax").text),
            int(xmlbox.find("ymax").text),
        )
        f.write(" " + ",".join([str(a) for a in b]) + "," + str(cls_id))


if __name__ == "__main__":
    # print("Called as the main function")
    main()
