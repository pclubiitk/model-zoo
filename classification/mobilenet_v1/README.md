# MOBILE NET v1 (TENSORFLOW)
### BASIC OVERVIEW: -
##### I have implemented the mobile net v1 model on the Fashion MNIST dataset.

#
#### 	MULTICLASS IMAGE CLASSIFICATION MODEL USING FASHION MNIST DATASET: -

- 	The training set contains 60000 images of 28x28 shape each and the test set contains 10000 images of 28x28 shape each. Both the sets contain images belonging to 10 classes.
- 	The architecture consists of two convolution layers which perform depthwise separable convolutions. Each convolution layer is followed by a maxpooling layer. 
- 	•	Finally, a flattening layer and a fully connected layer has been added and an output layer is added using the “SparseCategoricalCrossentropy” loss function.
- 	•	After 10 epochs of training, the training accuracy and training loss were 93% and 0.1790 respectively. The corresponding quantities for the test set were 90% and 0.2968 respectively.
#
#### MODEL ARCHITECTURE: -

##### The model architecture contains the basic features of the mobile net-v1 paper. Instead of standard convolution layers, two depthwise separable convolution layers have been added. Each convolution layer is followed by a max pooling layer.
#
#### IMPLEMENTATION: -
##### Run all the cells.
#
#### SOURCES: -
- 	https://arxiv.org/abs/1704.04861
- 	https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
