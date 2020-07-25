# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import seaborn as sns
from sklearn.metrics import confusion_matrix
import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
base_skin_dir=os.path.join('..','input')
test_csv ="../input/siim-isic-melanoma-classification/test.csv"
train_csv="../input/siim-isic-melanoma-classification/train.csv"
train_set=pd.read_csv(os.path.join(base_skin_dir,train_csv))
path=[]
for dirname, _, filenames in os.walk('../input/siim-isic-melanoma-classification/jpeg/train/'):
    for filename in filenames:
           path.append(dirname+filename)

train_set['path']=path
#print(df.head())
#df.isnull().sum()
train_set['sex'].fillna("Unknown",inplace=True)
train_set['age_approx'].fillna((df['age_approx'].mean()),inplace=True)
train_set['anatom_site_general_challenge'].fillna("Unknown",inplace=True)
train_set.drop_duplicates(keep=False,inplace=True)
#df.isnull().sum()
#print(df.dtypes)

%matplotlib inline
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
train_set['benign_malignant'].value_counts().plot(kind='bar', ax=ax1)
train_set['anatom_site_general_challenge'].value_counts().plot(kind='bar', ax=ax1)
train_set['sex'].value_counts().plot(kind='bar')

from PIL import Image
#df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((512,512))))
train_set['image'] = train_set['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train_y_test= train_test_split(features,target,test_size=0.2,random_state=10000)
x_test=x_test.reshape(x_train.shape[0],*(100,100,3))
x_train=