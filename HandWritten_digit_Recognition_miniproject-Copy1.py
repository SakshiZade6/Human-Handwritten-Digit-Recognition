#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import matplotlib.pyplot as plt

import keras

from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout


# In[27]:


# Get the data and preprocess it


# In[28]:


mnist.load_data


# In[29]:


(X_train,y_train),(X_test,y_test)=mnist.load_data()

X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[30]:


plt.imshow(X_train[0])


# In[31]:


plt.imshow(X_train[0],cmap='binary')


# In[32]:


def plot_input_img(i):
    plt.imshow(X_train[i],cmap='binary')
    plt.title(y_train[i])
    plt.show()
    


# In[33]:


for i in range(10):
    plot_input_img(i)
    


# In[34]:


# Pre-process the image

# Normalising the image 
X_train=X_train.astype(np.float32)/255
X_test=X_test.astype(np.float32)/255


# In[35]:


X_train.shape


# In[36]:


# Pre-process the image

# Normalising the image 
X_train=X_train.astype(np.float32)/255
X_test=X_test.astype(np.float32)/255

#Reshape or Expand the size of image to(28,28)
X_train=np.expand_dims(X_train,-1)
X_test=np.expand_dims(X_test,-1)


# In[37]:


X_train.shape


# In[38]:


# converting to one hot vector
y_train = keras.utils.to_categorical(y_train)

y_test = keras.utils.to_categorical(y_test)


# In[39]:


y_train


# In[40]:


# Model Building
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (28,28,1), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(10, activation="softmax"))


# In[41]:


model.summary()


# In[42]:


model.compile(optimizer= 'adam', loss = keras.losses.categorical_crossentropy, metrics=['accuracy'] )


# In[43]:


#callbacks

from keras.callbacks import EarlyStopping, ModelCheckpoint

#EarlyStopping 

es = EarlyStopping(monitor='val_acc', min_delta=0.01, patience= 4, verbose= 1)

#Model Checkpoint

mc = ModelCheckpoint("./bestmodel.h5",monitor="val_acc",verbose= 1, save_best_only= True)

cb = [es,mc]


# In[44]:


# Model Training 

his = model.fit(X_train, y_train, epochs=50, validation_split= 0.3 )


# In[45]:


his = model.fit(X_train, y_train, epochs=5, validation_split= 0.3, callbacks = cb)


# In[46]:


model.save('bestmodel.h5')


# In[47]:


model_S = keras.models.load_model('bestmodel.h5')


# In[48]:


model_S.summary()


# In[49]:


score = model_S.evaluate(X_test, y_test)

print(f" the model accuracy is{score[1]} ")


# In[ ]:




