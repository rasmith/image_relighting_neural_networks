import numpy as np
import sys

class ModelMaker:
  def __init__(self, light_dim, num_hidden_nodes):
    self.pixel_dim = 2 # i, j normalized to [0, 1]
    self.light_dim = light_dim # 1D, 2D, or 3D, normlized to [-1, 1]
    self.color_dim = 3 # r, g, b, normalized to [-1, 1]
    self.input_dim  = self.pixel_dim + self.light_dim + self.color_dim
    self.num_hidden_nodes = num_hidden_nodes

  def set_checkpoint_file(self, file_name):
    self.checkpoint_file = file_name

  def compile(self):
    pass

  def reset(self):
    pass

  def train(self, train_data, train_labels, batch_size, verbose = 0):
    pass
    return score


  def load_weights(self):
    pass

  def predict(self, test_data, batch_size):
    pass

