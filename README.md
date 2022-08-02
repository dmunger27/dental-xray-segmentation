# Semantic Segmentation with Dental X-Rays

This project is part of the ADS-599 course in the Applied Data Science program at the University of San Diego.

-**Project Status: Completed**

## Collaborators

* Dallin Munger
* Jack McCullers
* Tristan Demond

## Project Introduction

The goal of this project is to build a effective models for identifying and segmenting teeth and dental abnormalities in panoramic dental x-rays. The motivation behind this project is to improve outcomes and patient experience at dental offices. 

### Methods Used

* Machine Learning
* Computer Vision
* Convolutional Neural Network
* ResNet50v2
* U-Net
* Image Preprocessing
* Cloud Storage 

### Technologies and Libraries

* Python
* Tensorflow
* Keras
* AWS S3

### Project Organization

    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── Project Code       <- The data exploration, preprocessing, and modeling code used in this project.
    |
    └── Flask App          <- Folder for code and files used to deploy model through Flask.


## Project Description

This project utilized data from the Tufts Dental Database. Since the database requires permission to use, the data has not been stored in the repository. However, it can be requested and downloaded at http://tdd.ece.tufts.edu/. The data consists of 1000 panoramic dental-xrays, and for this project 3 individual datasets were used. One contains original radiograph images, one contains expert annotated teeth masks, and one contains expert annotated abnormality masks. The data were stored in AWS S3 prior to analysis. Google Colab was used due to its GPU capabilities, making the deep learning portion of this project possible.

Images were visualized and the attributes of each image were analyzed. For example, it was found that all images had the same size. In order to prepare the data for modeling, images needed to be cropped to focus on the teeth and resized for quicker processing. The color scale of the masks also had to be set to grayscale, and the original radiographs set to RGB. Data augmentation also had to be done to increase the number of data points on which to train the model, especially for the abnormality data. Due to the high class imbalance, with most of the mask images being background, it was difficult for the model to learn and recognize abnormalities in the x-rays. The model used was a U-Net model with a ResNet50v2 decoder. The initial training round for both models froze tha weights in the decoder layer and set a higher learning rate with an Adam optimizer. The second training round unfroze the weights with a low learning rate to tweak the model.

While the teeth model performed well, the abnormality model struggles to recognize dental abnormalities. This comes down to a few different issues. One is the lack of data. 1000 images is not a lot to train on, and even with augmentation techniques, the overall number of images to learn from was low. The abnormalities are also difficult to distinguish in the image and represent a minority of the overall image. 
