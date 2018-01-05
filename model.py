import keras
import numpy as np
import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K

class ModelMaker:
  def __init__(self, light_dim, num_hidden_nodes):
    self.pixel_dim = 2 # i, j normalized to [0, 1]
    self.light_dim = light_dim # 1D, 2D, or 3D, normlized to [-1, 1]
    self.color_dim = 3 # r, g, b, normalized to [-1, 1]
    self.input_dim  = self.pixel_dim + self.light_dim + self.color_dim
    self.num_hidden_nodes = num_hidden_nodes
    self.checkpoint_file = 'model.hdf5'
    self.model = Sequential()
    self.model.add(Dense(self.num_hidden_nodes, \
                       activation = 'tanh', \
                       input_dim = self.input_dim))
    self.model.add(Dense(self.num_hidden_nodes));
    # output labels r, g, b
    self.model.add(Dense(self.color_dim))

  def set_checkpoint_file(self, file_name):
    self.checkpoint_file = file_name

  def compile(self):
    self.model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

  def reset(self):
    self.model.reset_states()

  def train(self, train_data, train_labels, batch_size, verbose = 1):
    model_checkpoint = ModelCheckpoint(self.checkpoint_file, monitor='loss')
    self.model.fit(train_data, train_labels,
      batch_size=batch_size, callbacks=[model_checkpoint], \
          epochs = 1, verbose=verbose)

  def load_weights(self):
    self.model.load_weights(self.checkpoint_file)

  def predict(self, test_data, target_data, batch_size):
    return self.model.predict(test_data, batch_size) 

