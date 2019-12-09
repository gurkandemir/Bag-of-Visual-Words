# Bag of Visual Words
Bag of visual words (BOVW) is commonly used in image classification. Its concept is adapted from information retrieval and NLP’s bag of words (BOW).

The general idea of bag of visual words (BOVW) is to represent an image as a set of features. Features consists of keypoints and descriptors. Keypoints are the “stand out” points in an image, so no matter the image is rotated, shrink, or expand, its keypoints will always be the same. And descriptor is the description of the keypoint. We use the keypoints and descriptors to construct vocabularies and represent each image as a frequency histogram of features that are in the image. From the frequency histogram, later, we can find another similar images or predict the category of the image.

In this assignment, we are expected to develop an image classifier based on Bag-of-Features model using Python. We are given a dataset which contains variable number of instances per class (There are 7 classes: City, Face, Greenery, Building, House Indoor, Office, Sea). The dataset is also divided into two as training and test. We are expected to train our classifier using the training image set and test it using the test image set.

## Implementation
1. Find training images path.
2. Read train images, and compute SIFT descriptors.
3. Find the dictionary.
4. Quantize features.
5. Classify test images.

## How to Run?

In order to run image classifier, execute the following from the command line:

>python3 BoW.py --train\_path [TRAIN\_PATH] --test\_path [TEST\_PATH] --no\_clusters [NO\_CLUSTERS] --kernel [KERNEL] 
    

WHERE
1. **TRAIN\_PATH** : Path of train dataset.
2. **TEST\_PATH** : Path of test dataset.
3. **NO\_CLUSTERS** : Number of clusters, default 50..
3. **KERNEL** : Type of kernel, linear or precomputed, default linear.
    
    For example, if you want to classify images with **precomputed** kernel, where train path is **dataset/train** and test path is **dataset/test** by choosing **100** cluster centers. You have to execute following command:

>python3 BoW.py --train\_path dataset/train --test\_path dataset/test --no\_clusters 100 --kernel precomputed
