# Show And Tell :  A Neural Image Caption Generator

### Contributed By:
[Akhil Agrawal](https://github.com/akhilagrawal1001/)

### Using the repo:
1. Clone the repository by using `git clone`
2. Download the required dataset ( `Flickr8k` ) from [here](https://www.kaggle.com/shadabhussain/flickr8k)
3. Extract the dataset, and put it into the repository folder
4. The repository contains various files, execute `model.py` for creating, training the model by using `python model.py`
   
   > 4.1 Update variables, that contain paths to different files and folders ( details mentioned in files ) in `data_process.py`, `encoder.py` .
5. For generating captions, execute `generate_caption.py` by using `python generate_caption.py`

   > 5.1 Update the path of image for which caption is to be generated in `generate_caption.py`

# Paper Summary:

Being able to automatically describe the content of an image using properly formed English sentences is a very challenging task, but it could have great impact, for instance by helping visually impaired people better understand the content of images on the web. The description should not only contain the objects present in the image, but also how these objects relate to each other, and the activities they are involved in. Hence a language is also required other than the visual understanding of the image.

This paper provides a joint model that takes an image I as input, and is trained to maximize the likelihood p(S|I) of producing a target sequence of words S = {S1, S2, . . .} where each word St comes from a given dictionary, that describes the image adequately.

The inspiration for this paper was recent advancements in Machine Language Translation, where we train a Recurrent Neural Network, based on `encoder` and `decoder`. The encoder extracts the information (`in a vector`) of the input sentence ( say in English language ), and then the decoder predicts the target sentence ( say in French language ) based on the encoder output vector.

![image](https://user-images.githubusercontent.com/80061490/126967830-9637189e-6e38-44b3-88ed-1240e55cef9f.png)

## Model:
In machine translation, state-of-art results can be directly obtained by maximizing the probability of correct translation, given the input sequence in _end to end_ fashion both at training and inference time. RNN's are used to encode variable length sequence inputs to a fixed dimension vector ( which stores the semantic information of the sequence ) and to decode this vector to generate the desired output sequence.

Following this approach, the captions can be decoded from the image as : 

![image](https://user-images.githubusercontent.com/80061490/126968821-5cdcae66-41bd-46b1-8390-77874d5b45b3.png)

where θ are the parameters of our model, I is an image, and S its correct transcription. Since S represents any sentence, its length is unbounded. Thus, it is common to apply the chain rule to model the joint probability over S0, . . . , SN , where N is the length of this particular example as :

![image](https://user-images.githubusercontent.com/80061490/126969126-4fec2718-476a-4a67-ba56-6018811410e7.png)

It is natural to model p(St|I, S0, . . . , St−1) with a Recurrent Neural Network (RNN), where the variable number of words we condition upon up to t − 1 is expressed by a fixed length hidden state or memory ht. We choose to implement LSTM ( Long-Short Term Memory ) net, which has its own state-of-art performance for such NLP problems.

### Encoder :
For image representation we use Convolutional Neural Networks. They are widely used and are currently state-of-art for object recognition and detection tasks. In this paper we will use pre-trained model InceptionV3 (paper can be found [here](https://arxiv.org/pdf/1409.4842.pdf)) to extarct various features of the images. We will neglect the last layer of the model ( which is softmax prediction used in image classification task ) because we need a dense vector of image storing all the information of the image.

   ![Screenshot 2021-07-26 153334](https://user-images.githubusercontent.com/80061490/126971524-c6e657b8-c54c-42a0-9a8c-fb4c7ebefac1.png)
   
In this implementation pre-trained, non-trainable model on InceptionV3 is used.

### Decoder :
For generating the desired output sequence, we use a particular form of Recurrent Neural Networks called Long-Short Term Memory (LSTM's) net.

![image](https://user-images.githubusercontent.com/80061490/126973672-d3b659a3-ea68-440a-bede-1b9ea113ee23.png)

The LSTM model is trained to predict each word of the sentence after it has seen the image as well as all preceding words as defined by p(St|I, S0, . . . , St−1) . The image is sent into the model as `initial_states` of LSTM. There are 2 initial_states in LSTM cell, hidden state ( used for softmaxe prediction ) and a cell state ( used for memorizing old data ).

#### Model Architecture :

![image](https://user-images.githubusercontent.com/80061490/126975139-5eb26f3c-3d32-4759-8a95-75d2f7e32401.png)


1. Encoding the image, to get a vector representation of the image.
2. Obtaining a dense vector from encoder output with dimensions matching the LSTM hidden states dimensions.
3. Initialising decoder's initial state as the dense vector containing image information.
  > During training, the input data and output data of decoder differs in one time-step, so that given a sequence, decoder should predict the next word in the sequence.

## Training:
This model has been trained on kaggle (notebook can be found [here](https://www.kaggle.com/notanidev/image-caption)), for 25 epochs, and it took approximately 30 seconds per epoch to train (using GPU).

![image](https://user-images.githubusercontent.com/80061490/126982894-a868c38f-88a4-47ff-9f0a-644421ad965e.png)

The loss function varies with epochs as follows:

![image](https://user-images.githubusercontent.com/80061490/126980052-9a5f92b4-4685-4b21-a5e0-8f71bdcd6fae.png)

## Results:
![543363241_74d8246fab](https://user-images.githubusercontent.com/80061490/126983714-c6268a64-05fd-4555-a33f-d5cff1153203.jpg)

```Predicted Caption : man wearing a red headband and a leather outfit standing outside```

![3683185795_704f445bf4](https://user-images.githubusercontent.com/80061490/126984177-5055866d-177a-4477-b70e-9ff839632e15.jpg)

```Predicted Caption : girl and a boy and a dog are walking next to a dog seen from a dock```

![3539767254_c598b8e6c7](https://user-images.githubusercontent.com/80061490/126984401-f37eac36-f6d0-4e02-8cfc-dc785f5d8e82.jpg)

```Predicted Caption : are another man on a boat```

## References :
1. [Original Paper](https://arxiv.org/pdf/1411.4555.pdf)
2. [Image captioning with keras](https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8)
3. [Image captioning with keras 2](https://medium.com/analytics-vidhya/image-captioning-with-tensorflow-2d72a1d9ffea)
4. [Image captioning in tensorflow](https://www.youtube.com/watch?v=uCSTpOLMC48&t=385s) 
5. [Andrew Ng Course on Coursera](https://www.coursera.org/programs/indian-institute-of-technology-kanpur-on-coursera-4adct/browse?currentTab=CATALOG&productId=W62RsyrdEeeFQQqyuQaohA&productType=s12n&query=deep+learning&showMiniModal=true)
