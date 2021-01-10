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

### Introduction

The increasing intensity of arable farming and widespread use of chemical herbicides has led to the development of herbicide resistant weed species which threaten the yield and production of crop species. If uncontrolled, yield losses can exceed 50% (Nakka et al., 2019). Weed species naturally contain a small number of plants that are more resistant to herbicides. Repeated applications of these chemicals lead to selection pressures, allowing only those more resistant plants to survive. Over time this causes an increase in the proportion of resistant weed plants, reducing the effectiveness of herbicides, reducing yields and increasing costs to farmers.

Weeds could simply be considered as plants that are in the wrong place. In agricultural environments, they compete with crop plants for nutrients, water and light causing decreases in quality and size of crops (Sardana et al, 2017). Some weeds are more competitive than others. There are some weed species that are not as competitive or harmful to the crop and may give more benefits by increasing biodiversity than the damage they cause.

Weeds will often grow in persistent non-uniform patches and so it is useful to be able to identify the region in which they are growing to allow improved cultural weed management practices or applications of herbicides (Lambert et al., 2019, Yu et al, 2019). Herbicides are usually applied at a consistent rate throughout a field. By applying these more selectively to individual weed patches, resources and costs can be saved as well as reducing the risk of crop damage, environmental consequences or pest resistance to chemicals (Partel et al, 2019). 

It is important that weeds can be distinguished from crops at any stage of growth as early identification will reduce the impact they have. To prevent the increase of resistant weed species, and continue the effective control of weeds, new measures are needed that are both integrated and more specific. Broad, blanket application of herbicides is one of the least effective methods in the long term. 

Machine learning approaches can be applied to these problems, in this analysis through the use of Convolutional Neural Networks (CNNs). CNNs are a type of neural network specific to image datasets and can be used for classification tasks. They allow the information an image contains to be inputted into a model, without the spatial features being lost which are critical for image recognition tasks. CNNs are made up of layers and end by applying a function to output a probability of the image being in each class.

#### Objectives
1. Train a deep learning image classification model that can identify between weed and crop plants.
2. Improve the model accuracy by tuning the hyperparameters of the model and algorithm.

### Methods

Keras is a deep learning framework for python, it provides a way to define and train deep learning models and will be used with Tensorflow in GoogleColab for this project. 

#### Data

A dataset containing images of 12 different plant species is the basis of this analysis. It was originally discovered via the Kaggle Plant Seedlings Classification Competition (https://www.kaggle.com/c/plant-seedlings-classification/data). The full set of images were subsequently downloaded from the original source of this data (Giselsson et al 2017). The 12 species in the data set are broken down into 9 weed species and 3 crop species, common to the UK.

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


### Results

### Discussion

### References

