# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 00:49:49 2019

@author: atidem
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# include data : !! you have to find mnist dataset from web and split them test.csv,train.csv 
train = pd.read_csv("train.csv")
train.shape

test = pd.read_csv("test.csv")
test.shape

y_train= train["label"]
X_train = train.drop(labels=["label"],axis=1)

# barchart visualize
plt.figure(figsize=(15,7))
sns.countplot(y_train,palette="icefire")
plt.title("number of digit")
y_train.value_counts()

## normalize 
X_train = X_train / 255.0 
test = test / 255.0

## Reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

## label encoding 
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train,num_classes=10)

## train test split
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=2)

## create model 
from sklearn.metrics import confusion_matrix
import itertools

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(filters=8,kernel_size=(5,5),padding="Same",activation="relu",input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=16,kernel_size=(3,3),padding="Same",activation="relu",input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.45))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.45))
model.add(Dense(10,activation="softmax"))

optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)

model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])

epochs = 10
batch_size = 250

## Data Augmentation
datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=0.5,
                             zoom_range=0.5,
                             width_shift_range=0.5,
                             height_shift_range=0.5,
                             horizontal_flip=False,
                             vertical_flip=False)
datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train,y_train,batch_size=batch_size),epochs=epochs,validation_data=(X_val,y_val),steps_per_epoch=X_train.shape[0]//batch_size)

## visualization 
plt.plot(history.history["val_loss"],label="validation_loss")
plt.title("test loss")
plt.xlabel("num of epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

## confusion matrix 
Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred,axis=1)
Y_true =np.argmax(y_val,axis=1)
confusion_mtx = confusion_matrix(Y_true,Y_pred_classes)
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(confusion_mtx,annot=True,linewidths=0.01,cmap="Blues",linecolor="gray",fmt=".1f",ax=ax)
plt.xlabel("predict")
plt.ylabel("true")
plt.show()