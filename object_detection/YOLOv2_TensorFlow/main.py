import matplotlib.pyplot as plt
import  cv2
%matplotlib inline
from model import *
from utils import *
import os
import time
import logging
import argparse
import numpy as np
import random
#Labels for VOC2012 dataset
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--iou', type=float, default=0.01)
    LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
            'bus',        'car',      'cat',  'chair',     'cow',
            'diningtable','dog',    'horse',  'motorbike', 'person',
            'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']
    train_image_folder = "VOCdevkit/VOC2012/JPEGImages/"
    train_annot_folder = "VOCdevkit/VOC2012/Annotations/"            
    train_image, seen_train_labels = parse_annotation(train_annot_folder,train_image_folder, labels=LABELS)
    print("N train = {}".format(len(train_image)))  

    # checking features in our dataset
    y_pos = np.arange(len(seen_train_labels))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.barh(y_pos,list(seen_train_labels.values()))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(seen_train_labels.keys()))
    ax.set_title("The total number of objects = {} in {} images".format(
        np.sum(list(seen_train_labels.values())),len(train_image)
    ))
    plt.show()
    # using kmeans to find anchor box
    wh = []
    for anno in train_image:
        aw = float(anno['width'])  # width of the original image
        ah = float(anno['height']) # height of the original image
        for obj in anno["object"]:
            w = (obj["xmax"] - obj["xmin"])/aw # make the width range between [0,GRID_W)
            h = (obj["ymax"] - obj["ymin"])/ah # make the width range between [0,GRID_H)
            temp = [w,h]
            wh.append(temp)
    wh = np.array(wh)
    print("clustering feature data is ready. shape = (N object, width and height) =  {}".format(wh.shape))
    kmax = 11
    dist = np.mean
    results = {}
    for k in range(2,kmax):
        clusters, nearest_clusters, distances = kmeans(wh,k,seed=2,dist=dist)
        WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]),nearest_clusters])
        result = {"clusters":             clusters,
                "nearest_clusters":     nearest_clusters,
                "distances":            distances,
                "WithinClusterMeanDist": WithinClusterMeanDist}
        print("{:2.0f} clusters: mean IoU = {:5.4f}".format(k,1-result["WithinClusterMeanDist"]))
        results[k] = result
    Nanchor_box = 4
    results[Nanchor_box]["clusters"]
    # from above line we get anchor box we will using that anchor box further
    _ANCHORS01 = np.array([0.08285376, 0.13705531,
                        0.20850361, 0.39420716,
                        0.80552421, 0.77665105,
                        0.42194719, 0.62385487])
    # now we will convert images to size that we required
    print("*"*30)
    print("Input")
    timage = train_image[5]
    for key, v in timage.items():
        print("  {}: {}".format(key,v))
    print("*"*30)
    print("Output")
    inputEncoder = ImageReader(IMAGE_H=416,IMAGE_W=416, norm=normalize)
    image, all_objs = inputEncoder.fit(timage)
    print("          {}".format(all_objs))
    plt.imshow(image)
    plt.title("image.shape={}".format(image.shape))
    plt.show()

    GRID_H,  GRID_W  = 13 , 13
    ANCHORS          = _ANCHORS01
    ANCHORS[::2]     = ANCHORS[::2]*GRID_W  
    ANCHORS[1::2]    = ANCHORS[1::2]*GRID_H  
    ANCHORS 
    BATCH_SIZE        = 200
    IMAGE_H, IMAGE_W  = 416, 416
    GRID_H,  GRID_W   = 13 , 13
    TRUE_BOX_BUFFER   = 50
    BOX               = int(len(ANCHORS)/2)

    generator_config = {
        'IMAGE_H'         : IMAGE_H, 
        'IMAGE_W'         : IMAGE_W,
        'GRID_H'          : GRID_H,  
        'GRID_W'          : GRID_W,
        'LABELS'          : LABELS,
        'ANCHORS'         : ANCHORS,
        'BATCH_SIZE'      : BATCH_SIZE,
        'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
    }

    train_batch_generator = SimpleBatchGenerator(train_image, generator_config,
                                                norm=normalize, shuffle=True)

    [x_batch,b_batch],y_batch = train_batch_generator.__getitem__(idx=3)

    print("x_batch: (BATCH_SIZE, IMAGE_H, IMAGE_W, N channels)           = {}".format(x_batch.shape))
    print("y_batch: (BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + N classes) = {}".format(y_batch.shape))
    print("b_batch: (BATCH_SIZE, 1, 1, 1, TRUE_BOX_BUFFER, 4)            = {}".format(b_batch.shape))

    iframe= 1
    check_object_in_grid_anchor_pair(iframe)
    # following line will let us show how our encoder work
    for irow in range(5,7):
        print("-"*30)
        check_object_in_grid_anchor_pair(irow)
        plot_image_with_grid_cell_partition(irow)
        plot_grid(irow)
        plt.show()
    # now calling our model and adding some more features to it    
    IMAGE_H, IMAGE_W  = 416, 416
    GRID_H,  GRID_W   = 13 , 13
    TRUE_BOX_BUFFER   = 50
    BOX               = int(len(ANCHORS)/2)
    CLASS             = len(LABELS)
    ## true_boxes is the tensor that takes "b_batch"
    model, true_boxes = define_YOLOv2(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS, 
                                    trainable=False)
    model.summary()    
    path_to_weight = "yolov2.weights" 
    weight_reader = WeightReader(path_to_weight)
    print("all_weights.shape = {}".format(weight_reader.all_weights.shape))
    nb_conv = 22
    model = set_pretrained_weight(model,nb_conv, path_to_weight)
    # making 
    
        
    layer   = model.layers[-4] # the last convolutional layer
    initialize_weight(layer,sd=GRID_H*GRID_W)  


    # checking our loss function involved in the model
    LAMBDA_NO_OBJECT = 1.0
    LAMBDA_OBJECT    = 5.0
    LAMBDA_COORD     = 1.0
    LAMBDA_CLASS     = 1.0

    true_boxes = tf.Variable(np.zeros_like(b_batch),dtype="float32")
    loss_tf    = custom_loss(y_batch_tf, y_pred_tf) 
    loss = loss_tf
    loss.numpy()

    # training our model
    
    train_batch_generator = SimpleBatchGenerator(train_image, generator_config,
                                                norm=normalize, shuffle=True)
    CLASS             = len(LABELS)
    model, true_boxes = define_YOLOv2(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS, 
                                    trainable=False)
    nb_conv        = 22
    model          = set_pretrained_weight(model,nb_conv, path_to_weight)
    layer          = model.layers[-4] # the last convolutional layer
    initialize_weight(layer,sd=1/(GRID_H*GRID_W))


    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import SGD, Adam, RMSprop

    dir_log = "logs/"
    try:
        os.makedirs(dir_log)
    except:
        pass


    BATCH_SIZE   = args.batch_size
    generator_config['BATCH_SIZE'] = BATCH_SIZE

    early_stop = EarlyStopping(monitor='loss', 
                            min_delta=0.001, 
                            patience=3, 
                            mode='min', 
                            verbose=1)

    checkpoint = ModelCheckpoint('weights_yolo_on_voc2012.h5', 
                                monitor='loss', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='min', 
                                period=1)


    optimizer = Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=1e-08, decay=0.0)


    model.compile(loss=custom_loss, optimizer=optimizer)

    model.fit_generator(generator        = train_batch_generator, 
                        steps_per_epoch  = len(train_batch_generator), 
                        epochs           = args.epochs, 
                        verbose          = 1,
                        #validation_data  = valid_batch,
                        #validation_steps = len(valid_batch),
                        callbacks        = [early_stop, checkpoint], 
                        max_queue_size   = 3)


    #now training part is done we will check it on image

    imageReader = ImageReader(IMAGE_H,IMAGE_W=IMAGE_W, norm=lambda image : image / 255.)
    out = imageReader.fit(train_image_folder + "/2007_005430.jpg")
    X_test = np.expand_dims(out,0)

    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    y_pred = model.predict([X_test,dummy_array])

    netout         = y_pred[0]
    outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
    netout_scale   = outputRescaler.fit(netout)

    iou_threshold = args.iou
    final_boxes = nonmax_suppression(boxes,iou_threshold=iou_threshold,obj_threshold=obj_threshold)

    ima = draw_boxes(X_test[0],final_boxes,LABELS,verbose=True)
    figsize = (15,15)
    plt.figure(figsize=figsize)
    plt.imshow(ima); 
    plt.show()
def check_object_in_grid_anchor_pair(irow):
        for igrid_h in range(generator_config["GRID_H"]):
            for igrid_w in range(generator_config["GRID_W"]):
                for ianchor in range(generator_config["BOX"]):
                    vec = y_batch[irow,igrid_h,igrid_w,ianchor,:]
                    C = vec[4] ## ground truth confidence
                    if C == 1:
                        class_nm = np.array(LABELS)[np.where(vec[5:])]
                        assert len(class_nm) == 1
                        print("igrid_h={:02.0f},igrid_w={:02.0f},iAnchor={:02.0f}, {}".format(
                            igrid_h,igrid_w,ianchor,class_nm[0]))
def normalize(image):
        return image / 255.
def initialize_weight(layer,sd):
        weights = layer.get_weights()
        new_kernel = np.random.normal(size=weights[0].shape, scale=sd)
        new_bias   = np.random.normal(size=weights[1].shape, scale=sd)
        layer.set_weights([new_kernel, new_bias])        
if __name__ == '__main__':
    main()        