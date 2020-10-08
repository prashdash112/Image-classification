import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread,imshow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,BatchNormalization,Flatten,Dense,Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array,array_to_img
from sklearn.metrics import confusion_matrix
import os
import itertools
import random 
import shutil
import glob

os.chdir('C:\\Users\\Prashant\\Desktop\\datasets\\finger\\')
os.makedirs(name='train')
os.makedirs(name='test')
os.makedirs(name='valid')

os.chdir('C:\\Users\\Prashant\\Desktop\\datasets\\finger\\train\\')
for i in ['0','1','2','3','4','5','6','7','8','9']:
    os.mkdir(i)
    
os.chdir('C:\\Users\\Prashant\\Desktop\\datasets\\finger\\valid\\')
for i in ['0','1','2','3','4','5','6','7','8','9']:
    os.mkdir(i)
    
os.chdir('C:\\Users\\Prashant\\Desktop\\datasets\\finger\\test\\')
for i in ['0','1','2','3','4','5','6','7','8','9']:
    os.mkdir(i)

os.chdir('C:\\Users\\Prashant\\Desktop\\datasets\\finger\\')
# Batch transfer
for i in ['0','1','2','3','4','5','6','7','8','9']:
    for file in random.sample(glob.glob('C:\\Users\\Prashant\\Desktop\\datasets\\finger\\'+ i +'\\*'),50):
        shutil.move(file,'C:\\Users\\Prashant\\Desktop\\datasets\\finger\\train\\'+ i)
    
    for file in random.sample(glob.glob('C:\\Users\\Prashant\\Desktop\\datasets\\finger\\'+ i +'\\*'),30):
        shutil.move(file,'C:\\Users\\Prashant\\Desktop\\datasets\\finger\\valid\\'+ i)
    
    for file in random.sample(glob.glob('C:\\Users\\Prashant\\Desktop\\datasets\\finger\\'+ i +'\\*'),20):
        shutil.move(file,'C:\\Users\\Prashant\\Desktop\\datasets\\finger\\test\\'+ i)
        
train_path='C:\\Users\\Prashant\\Desktop\\datasets\\finger\\train\\'
valid_path='C:\\Users\\Prashant\\Desktop\\datasets\\finger\\valid\\'
test_path='C:\\Users\\Prashant\\Desktop\\datasets\\finger\\test\\'

train_batch=ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)

test_batch=ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10,shuffle=False)

valid_batch=ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)

mobile = tf.keras.applications.mobilenet.MobileNet()
x=mobile.layers[-6].output
output=Dense(10,activation='softmax')(x)

model=tf.keras.Model(inputs=mobile.input, outputs=output)

for layer in model.layers[:-23]:
    layer.trainable=False
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batch,
            steps_per_epoch=len(train_batch),
            validation_data=valid_batch,
            validation_steps=len(valid_batch),
            epochs=10,
            verbose=2
)

test_labels = test_batch.classes
predictions = model.predict(x=test_batch, steps=len(test_batch), verbose=0)
cm =confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
