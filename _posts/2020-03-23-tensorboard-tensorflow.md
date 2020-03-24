---
comments: true
author: krishan
layout: post
categories: deeplearning
title: Using Tensorboard in Tensorflow-Keras (windows version)
---


```python
# https://www.tensorflow.org/install/pip
# !pip install tensorboard
# !pip install tensorflow-cpu
```


```python
# Load the TensorBoard notebook extension
%load_ext tensorboard
```


```python
import tensorflow as tf
import datetime
```


```python
import os 
log_dir = os.path.join('logs','fit', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
print(log_dir)


!powershell rm -Force -R logs


os.makedirs(log_dir, exist_ok=True)
!powershell dir logs\fit
```

    logs\fit\20200323-192702
    
    
        Directory: C:\Users\krkusuk\projects\tensorboard\logs\fit
    
    
    Mode                LastWriteTime         Length Name                          
    ----                -------------         ------ ----                          
    d-----        3/23/2020   7:27 PM                20200323-192702               
    
    
    


```python
mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = (28,28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10,activation='softmax')
    ])
```


```python
model = create_model()
model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
print(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)

model.fit(x = x_train,
         y = y_train,
         epochs = 5,
         validation_data = (x_test,y_test),
         callbacks = [tensorboard_callback])
```

    logs\fit\20200323-192702
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/5
    60000/60000 [==============================] - 17s 285us/sample - loss: 0.2205 - accuracy: 0.9350 - val_loss: 0.1192 - val_accuracy: 0.9634
    Epoch 2/5
    60000/60000 [==============================] - 16s 272us/sample - loss: 0.0975 - accuracy: 0.9704 - val_loss: 0.0766 - val_accuracy: 0.9756
    Epoch 3/5
    60000/60000 [==============================] - 16s 268us/sample - loss: 0.0693 - accuracy: 0.9784 - val_loss: 0.0724 - val_accuracy: 0.9796
    Epoch 4/5
    60000/60000 [==============================] - 16s 270us/sample - loss: 0.0541 - accuracy: 0.9832 - val_loss: 0.0684 - val_accuracy: 0.9783
    Epoch 5/5
    60000/60000 [==============================] - 16s 264us/sample - loss: 0.0433 - accuracy: 0.9855 - val_loss: 0.0664 - val_accuracy: 0.9799
    




    <tensorflow.python.keras.callbacks.History at 0x25143f06d88>



# Start tensorboard

## Issues and resolution
Observed that once tensorflow goes into a bad state, it throws problem everytime afterwards because 

1. It does not kill previous processes automatically
2. It uses previous states while starting the dashboard

Steps to mitigate the bad state:

1. kill all running tensorboard processes.
2. Clear previous tensorboard state.



If it times out in jupyter, then go to http://localhost:6006 in the browser and check


```python
! powershell "echo 'checking for existing tensorboard processes'"
! powershell "ps | Where-Object {$_.ProcessName -eq 'tensorboard'}"

! powershell "ps | Where-Object {$_.ProcessName -eq 'tensorboard'}| %{kill $_}"

! powershell "echo 'cleaning tensorboard temp dir'"
! powershell "rm $env:TEMP\.tensorboard-info\*"

! powershell "ps | Where-Object {$_.ProcessName -eq 'tensorboard'}"


%tensorboard --logdir="logs\fit" --host localhost #quotes are important in windows


! echo If it has timed out in jupyter, then go to http://localhost:6006 in the browser and check
```

    checking for existing tensorboard processes
    
    Handles  NPM(K)    PM(K)      WS(K)     CPU(s)     Id  SI ProcessName          
    -------  ------    -----      -----     ------     --  -- -----------          
         87       6      944       4060       0.03   1112   1 tensorboard          
    
    
    cleaning tensorboard temp dir
    


    ERROR: Timed out waiting for TensorBoard to start. It may still be running as pid 6284.


    If it has timed out in jupyter, then go to http://localhost:6006 in the browser and check
    

#### From tensorboard documentations:

A brief overview of the dashboards shown (tabs in top navigation bar):

1. The Scalars dashboard shows how the loss and metrics change with every epoch. You can use it to also track training speed, learning rate, and other scalar values.
2. The Graphs dashboard helps you visualize your model. In this case, the Keras graph of layers is shown which can help you ensure it is built correctly.
3. The Distributions and Histograms dashboards show the distribution of a Tensor over time. This can be useful to visualize weights and biases and verify that they are changing in an expected way.

Additional TensorBoard plugins are automatically enabled when you log other types of data. For example, the Keras TensorBoard callback lets you log images and embeddings as well. You can see what other plugins are available in TensorBoard by clicking on the "inactive" dropdown towards the top right.

## References:

1. Tensorboard quickstart

https://www.tensorflow.org/tensorboard/get_started

2. No dashboard active error:

https://stackoverflow.com/questions/47113472/tensorboard-error-no-dashboards-are-active-for-current-data-set


2. [Windows] tensorboard - needs to be started from same drive as logdir 

https://github.com/tensorflow/tensorflow/issues/7856

3. localhost refused to connect.

https://github.com/tensorflow/tensorboard/issues/2481
