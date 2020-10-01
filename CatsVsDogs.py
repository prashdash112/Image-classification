import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread,imshow
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Activation,Flatten,BatchNormalization,Conv2D,MaxPool2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random 
import glob
import warnings
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore',category=FutureWarning)
import pathlib
%matplotlib inline

filenames = os.listdir('C:\\Users\\Prashant\\Desktop\\datasets\\train')
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

os.chdir(path="C:\\Users\\Prashant\\Desktop\\datasets\\train\\")
if os.path.isdir('train\dogs') is False:
    os.makedirs('train\dogs')
    os.makedirs('train\cats')
    os.makedirs('test\dogs')
    os.makedirs('test\cats')
    os.makedirs('valid\dogs')
    os.makedirs('valid\cats')

for i in random.sample(glob.glob('cat*'),500):
    shutil.move(i,'train\cats')
for i in random.sample(glob.glob('dog*'),500):
    shutil.move(i,'train\dogs')
for i in random.sample(glob.glob('cat*'),100):
    shutil.move(i,'valid\cats')
for i in random.sample(glob.glob('dog*'),100):
    shutil.move(i,'valid\dogs')
for i in random.sample(glob.glob('cat*'),50):
    shutil.move(i,'test\cats')
for i in random.sample(glob.glob('dog*'),50):
    shutil.move(i,'test\dogs')
    
train_path='C:\\Users\\Prashant\\Desktop\\datasets\\train\\train'
test_path='C:\\Users\\Prashant\\Desktop\\datasets\\train\\test'
valid_path='C:\\Users\\Prashant\\Desktop\\datasets\\train\\valid'

train_batches =ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path,
                                                                                             target_size=(224,224),
                                                                                             classes=['cats','dogs'],
                                                                                             batch_size=10,
                                                                                             class_mode='binary')

test_batches =ImageDataGenerator( 
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path,
                                                                                             target_size=(224,224),
                                                                                             classes=['cats','dogs'],
                                                                                             class_mode='categorical',
                                                                                             batch_size=10,shuffle=False)
valid_batches =ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path,
                                                                                             target_size=(224,224),
                                                                                             classes=['cats','dogs'],
                                                                                             batch_size=10,class_mode='binary')
                                                                                             
def plotImages(images_arr):
    fig, axes = plt.subplots(2, 5, figsize=(17,17))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

model=Sequential([Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3),padding='same'),
                  BatchNormalization(),
                  MaxPool2D(pool_size=(2,2),strides=2),
                  Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
                  BatchNormalization(),
                  MaxPool2D(pool_size=(2,2),strides=2),
                  Flatten(),
                  Dense(units=200,activation='relu'),
                  BatchNormalization(),
                  Dropout(0.5),
                  Dense(units=1,activation='sigmoid')                  
                 ])
                 
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

total_validate=100
total_train=500
batch_size=10

history = model.fit_generator(
    train_batches, 
    epochs=20,
    validation_data=valid_batches,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size
)

final_df1=pd.DataFrame(history.history).plot(figsize=(10,5))
plt.grid(True)
plt.gca().set_ylim(0, 5.2) # set the vertical range to [0-1]

x=test_batches
prediction=model.predict(x,verbose=0)
r=prediction
print(np.round(r))

c_m=confusion_matrix(y_true=test_batches.classes,y_pred=r.reshape(100,))
c_k=c_m/36
print(c_k) # normalized confusion matrix
