#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:32:18 2020

@author: avantari
"""

import pandas as pd
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

dataset = pd.read_csv('./dataset/dataset.csv')
dataset = dataset.iloc[:,:-1]


target = pd.read_csv('./dataset/dataset.csv')
target = target.iloc[:,-1]

train, test = train_test_split(dataset,test_size=0.33)

train_scaled = minmax_scale(train, axis = 0)
test_scaled = minmax_scale(test, axis = 0)

ncol = train_scaled.shape[1]
X_train, X_test = train_test_split(train_scaled, train_size = 0.9, random_state = seed(2020))

encoding_dim = 64

input_dim = Input(shape = (ncol, ))

# Encoder Layers
encoded1 = Dense(180, activation = 'relu')(input_dim)
encoded2 = Dense(140, activation = 'relu')(encoded1)
encoded3 = Dense(100, activation = 'relu')(encoded2)
encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)

# Decoder Layers
decoded1 = Dense(100, activation = 'relu')(encoded4)
decoded2 = Dense(140, activation = 'relu')(decoded1)
decoded3 = Dense(180, activation = 'relu')(decoded2)
decoded4 = Dense(ncol, activation = 'sigmoid')(decoded3)

# Combine Encoder and Deocder layers
autoencoder = Model(inputs = input_dim, outputs = decoded4)

# Compile the Model
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

autoencoder.fit(X_train, X_train, nb_epoch = 5000, batch_size = 32, shuffle = False, validation_data = (X_test, X_test))

encoder = Model(inputs = input_dim, outputs = encoded4)
encoded_input = Input(shape = (encoding_dim, ))


encoded_train = pd.DataFrame(encoder.predict(dataset))
encoded_train = encoded_train.add_prefix('feature_')
encoded_train['target'] = target


encoded_train.to_csv('train_encoded.csv', index=False)

# save model
encoder.save_weights('./encoder_weights')

# load model
encoder = Model(inputs = input_dim, outputs = encoded4)
encoder.load_weights('./encoder_weights')
encoder.predict(dataset)
