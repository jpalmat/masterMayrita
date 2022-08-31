#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Imports necesarios
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from skimage.transform import resize


# In[4]:


class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, data_frame, width, height, channels, batch_size=128, path_to_img="../data/images", shuffle=True):
        self.df = data_frame
        self.width = width
        self.height = height
        self.channels = channels
        self.batch_size = batch_size
        self.path_to_img = path_to_img
        self.shuffle = shuffle
        self.indexes = np.arange(len(data_frame.index))
        
    def __len__(self):
        return int(np.ceil(len(self.indexes)/self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #Inicializamos listas de datos
        X, Y = [], []
        
        for idx in indexes:
            x, y = self.get_sample(idx)
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)
    
    def get_sample(self, idx):
        df_row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.path_to_img, df_row["ImageID"]))
        image = image.resize((self.height, self.width))
        image = np.asarray(image)
        image2 = np.reshape(image, image.shape + (self.channels,))
        #image2.setflags(write=1)
        image2 = self.norm(image2)
        label = image2
        return image2, label
    
    def norm(self, image):
        image = image/255.0
        return image.astype(np.float32)


# In[ ]:




