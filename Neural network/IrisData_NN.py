#!/usr/bin/env python
# coding: utf-8

# ## Applying simple Neural Network model on Iris Data

# In[1]:


import numpy as np


# In[2]:


from sklearn.datasets import load_iris


# In[3]:


iris= load_iris()


# In[4]:


type(iris)


# In[5]:


print(iris.DESCR)


# In[6]:


X=iris.data
#X


# In[7]:


y=iris.target


# In[8]:


#class 0 ---> [1,0,0]
#class 1 ---> [0,1,0]
#class 2 ---> [0,0,1]
# to do this we are using Keras
from keras.utils import to_categorical


# In[9]:


y=to_categorical(y)


# In[10]:


print(y.shape)


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.33, random_state=42)


# In[13]:


from sklearn.preprocessing import MinMaxScaler # to scale value between 0 and 1


# In[14]:


scaler_object=MinMaxScaler()


# In[15]:


scaler_object.fit(X_train)


# In[16]:


scaled_X_train=scaler_object.transform(X_train)


# In[17]:


scaled_X_test=scaler_object.transform(X_test)


# In[18]:


#scaled_X_train


# In[19]:


# Building the network using keras 
from keras.models import Sequential # to get sequesces
from keras.layers import Dense # to make dense model i.e., no. of hidden layer greater than 2(simply saying)


# In[20]:


model= Sequential() #to creat Sequestial model
# to add layer to model (in this case Dense layer)
model.add(Dense(8,input_dim=4,activation='relu')) # taking '8 neuron' and input dimention 4=no. of features
model.add(Dense(8,input_dim=4,activation='relu')) # adding one more same layer as above
model.add(Dense(3,activation='softmax')) # taking '3 neuron' same as in outpur layer and using softmax here
#output should be close to [.2,.3,.5] A per value of y

# Now compiling it
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[21]:


# to get summary of model
print(model.summary())


# In[22]:


model.fit(scaled_X_train, y_train, epochs=150, verbose =2)
# verbose is just how much output infomation we want ex- verbose=0 give no information
#while verbose=2 with give information of accuracy


# In[23]:


model.predict(scaled_X_test) # it gives us probability of predict for each 3 output
#the highest probability we assume as answer


# In[24]:


print(y_test.shape)
print(scaled_X_test.shape)


# In[25]:


# to predict classes
model.predict_classes(scaled_X_test)


# In[26]:


print(y_test[0:5,:]) 


# In[27]:


# here y_test and prediction value is in different formet so we need to make them in same formate
predictions=model.predict_classes(scaled_X_test)
y_test.argmax(axis=1) #this different from y_test in case of format


# In[28]:


# now comparing predictions with y_test.argmax(axis=1)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
confusion_matrix(y_test.argmax(axis=1),predictions) # rows are actual and column are predicted value for each classes


# In[29]:


print(classification_report(y_test.argmax(axis=1),predictions))


# In[30]:


print(accuracy_score(y_test.argmax(axis=1),predictions))


# In[31]:


# Now to save this this model
#model.save('...filelocation/filename.h5')
model.save('E:/study material/Deep learning/neural network/IrisDataNN.h5')


# In[32]:


# to open this save model
from keras.models import load_model
new_model=load_model('E:/study material/Deep learning/neural network/IrisDataNN.h5')


# In[33]:


new_model.predict_classes(scaled_X_test)


# In[ ]:




