import keras
import numpy as np
import sys
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

  def compile(self):
    self.model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

  def reset(self):
    self.model.reset_states()

  def train(self, train_data, train_labels, batch_size):
    model_checkpoint = ModelCheckpoint(self.checkpoint_file, monitor='loss')
    model.fit(train_data, train_labels,
      batch_size=batch_size, callbacks=[model_checkpoint], verbose=1)

  def test(self):
    model.load_weights(checkpoint_file)
    score = model.evaluate(test_data, test_labels, verbose=1)
    return score
