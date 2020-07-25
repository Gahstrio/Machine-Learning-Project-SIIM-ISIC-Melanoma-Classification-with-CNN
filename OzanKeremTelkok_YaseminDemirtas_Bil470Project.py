import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models #tensorflow.keras is used for creating CNN model and training it.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
from PIL import ImageFile
import random
import os
def accessImage(path,resize):#Accessing image through given path. resize parameter does not function
                             # because default size is 512x512
    image = Image.open(path)
    image = image.resize((512, 512))
    image = np.array(image)
    return image
base_path = '/kaggle/input/siim-isic-melanoma-classification'# Paths for accessing data
train_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

model = models.Sequential()
model.add(layers.Conv2D(16, (8, 8), strides=(8,8) ,activation='relu', input_shape=(512, 512, 3)))#First layer takes input of (512x512x3)
                                                                                                # and creates a new matrix of (64x64x16)
model.add(layers.MaxPooling2D((2, 2)))#Max pooling takes an input of 64x64x16 and creates a new matrix of (32x32x16)
model.add(layers.Conv2D(32, (8, 8),strides=(2,2) ,activation='relu'))#Second layer takes input of 32x32x16 and creates a new matrix of size 13x13x32
model.add(layers.Conv2D(32, (8, 8), activation='relu'))#Third layer from (13x13x32)->(6x6x32)
model.add(layers.Flatten(input_shape=(6, 6,32)))# Matrix is flattened to evaluate a classification
model.add(layers.Dense(2,activation = 'softmax')) 
model.compile(optimizer='adam',
                  loss='binary_crossentropy',#At the beginning of the project we used default loss function and after
                                              # noticing that an overfit for benign cases occurs, we changed it to binary_crossentropy
                                              # this change cause significant performace optimization for predicting malignant cases
                  metrics=['accuracy'])
model.summary()
trainF = pd.read_csv(os.path.join(base_path, 'train.csv'))# reading csv file which contains names of images and other features
breakFlag = 0
for y in range(350): #training CNN with a batch of 100 samples. We chose 100 samples because of memory limitations of kaggle machines.
                     # model.fit() function actually can change size of batches and epochs, but because of memory limitations it is not
                     # usable.
    lastIndex = (y+1)*100
    if lastIndex > len(trainF['image_name']):#Breakpoint for iteration where number of samples is not direct multiple of 100.
        lastIndex = len(trainF['image_name'])-1 # If case sample is between i*100 and (i+1)*100, lastIndex is set to (i+1)*100
                                                # for staying in bounds and breakFlag is set to 1 to end iteration
        breakFlag = 1
    train = trainF[y*100:lastIndex]
    trainIm = train['image_name']
    train_images = trainIm.values.tolist()
    train_images = [os.path.join( i + ".jpg") for i in train_images]
    trImages = []
    for imname in train_images:
            img = accessImage(train_img_path+imname,[512,512])
            trImages.append(img)
    trImages = np.asarray(trImages)
    trImages = trImages/255.
    trTargets = np.asarray(train['target'])
    class_weight = {0: 2.,1: 10.,}# Different values for class_weight has attempted. {(2.,98.),(2.,70.),(2.,50.),(2.,25.),(2.,10.)}
                                  # purpose of this application is to find optimal weights for prediction accuracy
                                  # on a different set of attempts, resampling malignant cases for numbers of {5,10,15,20} is attempted.
                                  # purpose of this application is to compare prediction performanca of resampling method to using class weights
                                  # method. It has seen that using class_weights is more efficient ####
    if len(trImages)==0 or len(trTargets)==0:# unnecesary code part from debugging
        break
    history = model.train_on_batch(trImages, trTargets,class_weight = class_weight)
    if breakFlag == 1:
        break
del train
del trainF
del trainIm
del train_images
del trImages
del trTargets# for memory optimization, data unused in test section is freed from memory
             # this is how we solved the problem of machine restarting because of memory usage
             # out of bounds.
breakFlag = 0
testF = pd.read_csv(os.path.join(base_path, 'test.csv'))## Reading csv file which contains names of images and other features for test data
output = []
for y in range(150):
    lastIndex = (y+1)*100
    if lastIndex > len(testF['image_name']):# breakpoint of iteration where sample number is between i*100 and (i+1)*100
        lastIndex = len(testF['image_name'])
        breakFlag = 1
    test = testF[y*100:lastIndex]
    testIm = test['image_name']
    test_images = testIm.values.tolist()
    test_img = [os.path.join( i + ".jpg") for i in test_images]
    teImages = []
    for imname in test_img:
        teImages.append(accessImage(test_img_path+imname,[512,512]))
    teImages = np.asarray(teImages)
    teImages = teImages / 255.
    pred_y = model.predict_on_batch(teImages)# predicting on batched too becaue of memory issues
    for (imname,elements) in zip(test_images,pred_y):
        output.append([imname,elements[1]])# predictions for test data is hold in output(list object)
    if breakFlag == 1:
        break
del testF
del test
del testIm
del test_images
del teImages# freeing unused data in next part from memory
f = open("submission.csv", "w+")
f.write('image_name,target'+os.linesep)
for element in output:
    f.write(element[0]+','+str(element[1]) + os.linesep)
f.close()
    
