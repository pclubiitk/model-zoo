MOBILE NET v1 (TENSORFLOW)
					(MODEL ZOO)


BASIC OVERVIEW: -
I have implemented the mobile net v1 model on two datasets. 
i.	Binary image classification dataset: - the model classifies the images into two categories namely, cats and dogs. (model1.ipynb)
ii.	Multiclass Image Classification using Fashion MNIST dataset. (model2.ipynb)

1.	BINARY IMAGE CLASSIFICATION MODEL: -
•	The dataset consists of 10000 images. It has been divided into two sets- a training set (8000 images belonging to 2 classes) and a test set (2000 images belonging to 2 classes).
•	The model architecture has been kept pretty simple and straightforward. There are two convolution layers which apply depthwise separable convolutions on the input, each followed by a max pooling layer.
•	The exact parameters can be viewed in the implemented Jupyter notebook file.
•	Finally, a flattening layer and a fully connected layer is added and an output layer is added using the “binary-crossentropy” loss function.
•	After 25 epochs of training, the training accuracy and training loss were 83% and 0.3670 respectively. The test accuracy and test loss were 78% and 0.4610 respectively.
•	At the last, a single prediction has been performed based on the training.

2.	MULTICLASS IMAGE CLASSIFICATION MODEL USING FASHION MNIST DATASET: -
•	The training set contains 60000 images of 28x28 shape each and the test set contains 10000 images of 28x28 shape each. Both the sets contain images belonging to 10 classes.
•	The architecture consists of two convolution layers which perform depthwise separable convolutions. Each convolution layer is followed by a maxpooling layer. 
•	Finally, a flattening layer and a fully connected layer has been added and an output layer is added using the “SparseCategoricalCrossentropy” loss function.
•	After 10 epochs of training, the training accuracy and training loss were 93% and 0.1790 respectively. The corresponding quantities for the test set were 90% and 0.2968 respectively.



MODEL ARCHITECTURE: -
The model architecture contains the basic features of the mobile net-v1 paper. Instead of standard convolution layers, two depthwise separable convolution layers have been added. Each convolution layer is followed by a max pooling layer.

IMPLEMENTATION: -

Download the dataset and use the appropriate directory/folder name while using the ImageDataGenerator function or loading image files. Then, run all the cells.

SOURCES: -

1.	https://arxiv.org/abs/1704.04861

2.	https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
