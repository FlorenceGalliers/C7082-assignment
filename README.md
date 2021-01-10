# Weed or Crop?
## Training a deep learning model to identify between images of weed and crop species.

[Github Repository](https://github.com/FlorenceGalliers/C7082-assignment) 

[Publicly Accessible Webpage](https://florencegalliers.github.io/C7082-assignment/)

**Scripts** within repository
- [Renaming Files](scripts/renaming-files.ipynb)
- [Image Organisation](scripts/image-organisation.ipynb)
- [Model Selection](scripts/model-selection.ipynb)
- [Final Model](final-model.ipynb) - This contains the code used for the creation of the final model

**Data**
- [All Data](all-data) A full collection of images used, split into 12 classes
- [New Data](new-data) Images from above file, split into three groups: train, validation and test, each containing 12 folders, one for each class.
- [GBIF Data](GBIF-data) Images downloaded from [GBIF](https://www.gbif.org/), split into 12 folders, one for each class.

**Graphs** contains outputs from model selection and figures contained in the final report
- [Final Model Accuracy Graph](graphs/final-model-acc.png)
- [Final Model Loss Graph](graphs/final-model-cost.png)

## Introduction

The increasing intensity of arable farming and widespread use of chemical herbicides has led to the development of herbicide resistant weed species which threaten the yield and production of crop species. If uncontrolled, yield losses can exceed 50% (Nakka et al., 2019). Weed species naturally contain a small number of plants that are more resistant to herbicides. Repeated applications of these chemicals lead to selection pressures, allowing only those more resistant plants to survive. Over time this causes an increase in the proportion of resistant weed plants, reducing the effectiveness of herbicides, reducing yields and increasing costs to farmers.

Weeds could simply be considered as plants that are in the wrong place. In agricultural environments, they compete with crop plants for nutrients, water and light causing decreases in quality and size of crops (Sardana et al, 2017). Some weeds are more competitive than others. There are some weed species that are not as competitive or harmful to the crop and may give more benefits by increasing biodiversity than the damage they cause.

Weeds will often grow in persistent non-uniform patches and so it is useful to be able to identify the region in which they are growing to allow improved cultural weed management practices or applications of herbicides (Lambert et al., 2019, Yu et al, 2019). Herbicides are usually applied at a consistent rate throughout a field. By applying these more selectively to individual weed patches, resources and costs can be saved as well as reducing the risk of crop damage, environmental consequences or pest resistance to chemicals (Partel et al, 2019). 

It is important that weeds can be distinguished from crops at any stage of growth as early identification will reduce the impact they have. To prevent the increase of resistant weed species, and continue the effective control of weeds, new measures are needed that are both integrated and more specific. Broad, blanket application of herbicides is one of the least effective methods in the long term. 

Machine learning approaches can be applied to these problems, in this analysis through the use of Convolutional Neural Networks (CNNs). CNNs are a type of neural network specific to image datasets and can be used for classification tasks. They allow the information an image contains to be inputted into a model, without the spatial features being lost which are critical for image recognition tasks. CNNs are made up of layers and end by applying a function to output a probability of the image being in each class.

### Objectives
**1. Train a deep learning image classification model that can identify between weed and crop plants.**

**2. Improve the model accuracy by tuning the hyperparameters of the model and algorithm.**

## Methods

Keras is a deep learning framework for python, it provides a way to define and train deep learning models and will be used with Tensorflow in GoogleColab for this project. 

### Data

A dataset containing images of 12 different plant species is the basis of this analysis. It was originally discovered via the Kaggle Plant Seedlings Classification Competition (https://www.kaggle.com/c/plant-seedlings-classification/data). The full set of images were subsequently downloaded from the original source of this data (Giselsson et al 2017). The 12 species in the data set are broken down into 9 weed species and 3 crop species, a sample image from each data set is shown in Figure 1.

![Sample Images of Each Class](https://github.com/FlorenceGalliers/C7082-assignment/blob/main/graphs/sample-images.png)
Figure 1: A sample image from each species class in the dataset.


Images are PNG files in a RGB format, the original images are all different sizes. Images will all be resized during the defining of the model generator. Different image sizes were explored from 150x150 upwards, but the size 299x299 was chosen as it yielded more accurate results than smaller images. 

A second set of 290 images were downloaded from GBIF (https://www.gbif.org/), with 20 to 25 images per class. This will later be used as a secondary test set for the final model.

The data is split into three groups: training, validation and test. The validation and test sets each contain 25 images of each species, with the remaining images all being used for training. In all of the subgroups, images are split into 12 folders which act as classes.

Each species has been given a three letter abbreviation (Table 1).


| Plant Name    | Weed or Crop? | Abbreviation | Total Images |
| ------------- |:-------------:| ------------:| -----------: |
| Black Grass, *Alopecurus myosuroides*  | Weed          | bgs          | 309           | 
| Charlock, *Sinapis arvensis* | Weed | chk | 452 |
| Cleavers, *Galium aparine* | Weed | cls | 335 |
| Common Chickweed, *Stellaria media* | Weed | cwd | 713 |
| Cranesbill, *Gerenium pusillum* | Weed | cbl | 576 |
| Loose Silky Bent, *Apera spica-venti* | Weed | lsb | 762 |
| Scentless Mayweed, *Matricaria inodora* | Weed | smw | 607 |
| Sheperd's Purse, *Capsella bursa-pastoris* | Weed | shp | 274 |
| Fat Hen, *Chenopodium album* | Weed | fhn | 538 |
| Wheat, *Tricicum aestivum* | Crop | wht | 253 |
| Maize, *Zea mays* | Crop | mze | 257 |
| Sugar Beet, *Beta vulgaris* | Crop | sbt | 463 |


Table 1: Summary of image information

### Image Preprocessing
A CNN will only accept floating point tensors as inputs. Tensors are simply containers for data. The images contain three channels of information in the RBG format, these are numbers between 0 and 255. To format these into tensors they need to be decoded into an RGB grid of pixels and converted into floating point tensors. This is done using the `keras.preprocessing.image` function `ImageDataGenerator`. This function also rescales the tensors so the values are between [0, 1].

It is important that the training and test sets are preprocessed in the same way to ensure the model accuracy can remain high on the test set. The model will be trained using supervised learning where the training dataset contains the class labels.

### Transfer Learning
In an image classification convnet, transfer learning is carried out by taking the convolutional base of a previously trained network, running the new data through it and then training a new densely connected classifier on top of the output. The learned representations from the previously trained network are usually generic and so are able to be reused on a new problem, it is the classifier layer that becomes more specific. Transfer learning models can be more complex than newly created CNNs and so they allow for more accurate classification.

There are different pre-trained networks that can be used, some examples are VGG16, Xception, ResNet50, InceptionV3 and MobileNet. For this problem the **Xception** network will be used (Chollet, 2017). Xception has less parameters than other models but has been shown to have higher accuracies. The Xception network is based on depthwise separated convolutional layers. The structure of this base is shown below (Figure 1). The weights learnt from training on the ‘ImageNet’ database were used in this model, and the base was frozen so it could not be trained further.

![Xception Model Architecture](https://github.com/FlorenceGalliers/C7082-assignment/blob/main/graphs/xception-model.png)
Figure 2: Xception model architecture (Chollet, 2017)

There are two types of hyperparameters to be considered. Firstly there are model hyperparameters which influence model selection, such as the number and width of hidden layers. Secondly the algorithm hyperparameters, these influence the speed and quality of the learning algorithm, for example learning rate of optimiser. Both types of hyperparameters were assessed and the method code file shows the creation of the optimal model that gave the highest validation accuracy.

The `Sequential()` function was used to define the model architecture. The pretrained convolutional base is imported from the Xception model and acts as the first ‘layer’ when defining the model, although it contains 14 blocks of layers. A new densely connected classifier is defined on top of this. Firstly a Global Max Pooling layer is used which downscales the feature maps of the inputs it receives. Then a dense layer, which learns global patterns followed by a batch normalisation layer that learns local patterns. Adding in a dropout layer next causes a random selection of units to be removed from the network, the value of 0.5 was chosen as this is near to the optimal for most models (Srivastava et al. 2014). Dropout layers help to reduce overfitting. This is followed by a further two dense and batch normalisation layers. The output layer is a dense later, with ‘softmax’ activation. The number of units in this layer should be equal to the number of classes in the data, in this case 12.

The next step is to compile the model using the `model.compile()` function. In this step the loss function, optimiser and metrics must be defined.

**Loss Function:** This is the quantity that will be minimised during training, it represents the measure of success. The loss function has to match the problem that is being solved. In this case it is a multi-class single-label classification problem, and so **categorical cross-entropy** is used as the loss function.

**Optimiser:** This determines how the network will be updated during training based on the loss function. It implements a specific variant of stochastic gradient descent.  This model uses the Adam optimiser which is an adaptive learning rate gradient descent. The goal of any optimiser is to calculate the weights that optimally minimise the loss function. The learning rate chosen to start with was 0.001.

**Metrics:** Training loss and validation loss can be monitored during model training to assess if there is under or overfitting. When the model's validation performance does not improve but the training performance continues to improve, there is overfitting in the model. As overfitting is undesirable, methods can be used to reduce the effects of this. The metric for evaluation is ‘accuracy’ as this is what will be assessed.

Eight versions of the model were created, with small changes each time with the hope of improving accuracy and minimising issues such as overfitting. A summary of the results of each model and the changes made each time are shown below. If a parameter is not changed, it remains the same as in the model previously created. Values given are approximate training and validation accuracy.

#### Model 1
- Image Size = 150 x 150 x 3
- Learning rate of optimiser = 0.001
- Batch size = 32
- 89%, 73%

The number of training sample images is relatively low in comparison to larger scale models and so overfitting was seen to begin with. The goal is to achieve a model that does not over or underfit. Different methods were used to try and reduce the effects of overfitting.

#### Model 2
- Learning rate increased to 0.005
- Introduce `reduce_lr` and `early_stop` callbacks
- Early stopping @ 31 epochs
- 88%, 73%

The first method used was the introduction of **callback** functions, although these are not strictly related to overfitting they are important in the creation of the model. Two callback functions were used in this one, `ReduceLROnPlateau()` and `Earlystopping()`. They both monitor the validation loss value.
- ReduceLROnPlateau causes adjustments in learning rate when the model performance is not improving. It is adjusted by multiplication by the factor 0.5. The ‘patience’ option in this function can be used to set the number of epochs that should take place with no improvement in performance before learning rate is decreased. It was set to 3 in the creation of this model. Minimum learning rate can also be defined, at 0.0005 here.
- EarlyStopping is a callback that stops the training of a model if no improvement in model performance is seen. It is useful because it saves time and computational energy if a model has stopped improving. The ‘patience’ option is also defined in this callback and was set to 10 epochs.

After these were added into the model, accuracy became more stable and the early fitting function was called in most cases with the training ending early.

#### Model 3
- Added data augmentation to training and validation data
- Early stopping @ 47 epochs
- 82%, 75%

Data augmentation generates more artificial training data from existing training samples, by augmenting the samples through random transformations. It is useful when the data set is small. In Keras the `ImageDataGenerator` function was used to augment images, this is the same function that was used to rescale images. There are many data augmentation options, however the idea of this model is to correctly distinguish between different plant species, so by changing the images too much or distorting them it will not help. The chosen augmentation options are only those that will keep the image in the correct shape to aid with correct identification of plants. Vertical and horizontal flipping and changes in brightness were the options of augmentation chosen. The brightness option is interesting as it may help the model work better on different images taken at different times of day. The brightness range of 50 to 150% was used.

#### Model 4
- Used pretrained weights from ‘imagenet’ for the base instead of those that are randomly generated
- Early stopping @ 47 epochs
- 82%, 76%

#### Model 5
- Changed Dense units in classifier 
  - From 100 units to 256
  - From 50 units to 128
- Early stopping @ 47 epochs
- 85%, 77%

#### Model 6
- Increased image size to 200 x 200 x 3
- Increase batch size from 32 to 64
- Early stopping @ 29 epochs
- 89%, 85%

#### Model 7
- Increased learning rate to 0.01
- Early stopping @ 37 epochs
- 91%, 85%

#### Model 8
- Increased image size to 299 x 299 x 3
- Early stopping @ 29 epochs
- 93%, 86%

After this model improvement process, the final model chosen was model 8, run with 50 epochs and a batch size of 64.

### Results of Final Model
The model showed a maximum training accuracy of **93%** and a maximum validation accuracy of **89%**. The training and validation accuracy were not far apart, and both continued increasing until the last epochs, this suggests overfitting was not causing a problem in this model. 
Model cost during training reduced to **0.186** and during validation was at a minimum value of **0.369**.

![Final Model Acc Graph](https://github.com/FlorenceGalliers/C7082-assignment/blob/main/graphs/final-model-acc.png)

Figure 3: Training and Validation Accuracy of Final Model

![Final Model Loss Graph](https://github.com/FlorenceGalliers/C7082-assignment/blob/main/graphs/final-model-cost.png)

Figure 4: Training and Validation Loss of Final Model


Testing this model on a test data set gave an accuracy score of **87.67%** and a loss value of **0.46**. This is very near to the validation accuracy, showing that that model performs well on unseen data.

### Discussion

Although the results yielded here are not at practically usable levels as they still misclassify plants, they show the potential that deep learning methods have in the future of agriculture. By accurately identifying weed species from images, these models could be integrated into machinery and used for more precise application of chemicals. Site specific weed control measures such as precision spraying have the advantage of greatly reducing the volume of herbicides needed (Gee and Denimal, 2020, Yu et al, 2019). This also has the benefit of reducing costs compared to manual hand spraying and reducing negative environmental impacts such as surface runoff and water pollution that come with uniform machine spraying of entire fields (Ofori and El-Gayar, 2020).

Precision agriculture is an increasingly popular method of integrating technology such as drones, robotics and artificial intelligence with agricultural practices (Ofori and El-Gayar, 2020).

The weights used in the final model were those trained on the ImageNet dataset, this is a  classification dataset with 10,000 object classes. By using weights from a dataset that is of something more similar to different plant species, accuracy may have been improved. By unfreezing these base layers and carrying out fine tuning of the model, a higher accuracy may have been obtained. Fine tuning optimises the weights of both the classifier and some of the base and leads to increases in accuracy (Siddiqi, 2019)

Low accuracy suggests that there is a high chance misclassification could occur. This may just be between different weed species, rather than between ‘weed’ or ‘crop’. A different approach that was not taken here would be to have just two classes of data, one containing weed species and one crop species. This could produce more accurate results as there would only be two classes. In practice for weed control purposes, for example if this technology was used within a sprayer, it would only need to know whether to spray a plant or not. This suggests that simplification of the model classes may be possible.

Another approach would be to develop crop specific models, for example one for use in a field where wheat is grown and a different one for maize fields. This would allow higher specificity in training of just one crop species against the weed species. The model in theory would only need to learn crop or not crop. It would be unusual to grow a combination of different crops in one field, therefore it makes sense to separate them out.

The detection of weed cover is an active area of research and the Identification of weed and crop species is used in the creation of weed cover maps. Partel et al (2019) put forward a method combining a CNNs, a weed mapping system and a precision sprayer in order to reduce the quantity of herbicides applied. Utilising real time image capture this shows the potential of these systems to aid in reducing chemical inputs in a commercially feasible method. Weed mapping using satellite imagery has seen mixed results but image quality and cloud cover have reduced the accuracy of this method. Using methods based closer to the ground, image quality and precision could be improved.

Past research has shown that the application of an image classification model trained in one environment performs inaccurately when transferred to a new environment (Lambert et al, 2019). To test this theory, the model created above was tested using images collected from another source (GBIF). Accuracy was greatly decreased to 23.10% and model loss increased to 5.54, this is in agreement with previous research showing that models trained with one set of images perform poorly with completely new images. Perhaps by combining these images with the original image set, and training the model with a shuffled combination, the accuracy could be improved. However, when then tested with an alternative new dataset, the decrease in accuracy may still occur.

Image quality is low in some of the training images due to the variation in size and the artificial backgrounds of the images makes the model difficult to transfer to other groups of images. All images are taken in one place and the background noise caused by trays, measuring tapes and stones may have led to lower accuracy. 

Future models created with this data set should look to balance the number of images in each class, and combine new images into the training so the model is more transferable. Cross validation for hyperparameter searching is something else that could be explored to see if the optimal values are being used. This model creation had no detection component, and it may be that the model is recognising other things in the images such as stones or trays, rather than the plants themselves. By training a model to detect only the plant, model improvement may be seen.

### References

Chollet, F. 2017. Xception: Deep learning with depth wise separable convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1251-1258

Gee, C. and Denimal, E. 2020. RGB Image-Derived Indicators for Spatial Assessment of the Impact of Broadleaf Weeds on Wheat Biomass. Remote Sensing, 12(18), pp.2982. https://doi.org/10.3390/rs12182982 	 	

Giselsson, T.M., Jørgensen, R.N., Jensen, P.K., Dyrmann, M. and Midtiby, H.S. 2017. A public image database for benchmark of plant seedling classification algorithms. arXiv, pp.arXiv-1711.

Lambert, J.P., Childs, D.Z. and Freckleton, R.P. 2019. Testing the ability of unmanned aerial systems and machine learning to map weeds at subfield scales: a test with the weed Alopecurus myosuroides (Huds). Pest Management Science, 75, pp.2283-2294. https://doi.org/10.1002/ps.5444

Nakka, S., Jugulam, M., Peterson, D., Asif, M. 2019. Herbicide resistance: Development of wheat production systems and current status of resistant weeds in wheat cropping systems. The Crop Journal, 7(6), pp.750-760. https://doi.org/10.1016/j.cj.2019.09.004.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R. 2014. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15, pp.1929-1958

Ofori, M., and El-Gayar, O. 2020. Towards Deep Learning for Weed Detection: Deep Convolutional Neural Network Architectures for Plant Seedling Classification. AMCIS 2020 Proceedings. 3.

Partel, V., Kakarla, S.C., Ampatzidis, Y. 2019. Development and evaluation of a low-cost and smart technology for precision weed management utilizing artificial intelligence. Computers and Electronics in Agriculture, 157, pp.339-350,

Siddiqi, R. 2019. Effectiveness of Transfer Learning and Fine Tuning in Automated Fruit Image Classification. In Proceedings of the 2019 3rd International Conference on Deep Learning Technologies (ICDLT 2019). Association for Computing Machinery, New York, USA, 91–100. DOI:https://doi.org/10.1145/3342999.3343002

Sardana, V., Mahajan, G., Jabran, K., Chauhan, B.S. 2017. Role of competition in managing weeds: An introduction to the special issue. Crop Protection, 95, pp.1-7,

Yu, J., Sharpe S.M., Schumann, A.W., Boyd, N.S. 2019. Deep learning for image-based weed detection in turfgrass. European Journal of Agronomy, 104, pp.78-84.
