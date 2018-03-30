import sys
import time
import glob
import os
from scipy import misc
import numpy as np
from multiprocessing import Pool
from model import ModelMaker
import tensorflow as tf
from keras import backend as K
import matplotlib.image as mpimg

from cluster import *

def init(models_dir, img_dir):
  assert os.path.exists(models_dir)
  if not os.path.exists(img_dir):
    os.mkdir(img_dir) 
  return (models_dir, img_dir)

dirname = sys.argv[1]

(model_dir, img_dir, width, height, num_images, ensemble_size,  max_levels, \
    sampled, assignments, average_img) = 
  config.load_cfg(dirname)

init(model_dir, img_dir)
img = np.ndarray((height, width), dtype = order='C')

# Need to get [[L, i, i, i, i, i],
#              [L, i, i, i, i, i], ...
# to [x, y, j, r, g, b], ...
# load model file, execute
               
# TODO - code this up               
test_data, ensemble_data = kmeans2d.get_ensemble_assignments(\
    assignments, average_img)

# predict image pixels
for data in ensemble_data:
  ensemble_id, level, start, end =  data
  batch_size = start - end + 1
  with tf.device('/cpu:0'):
    model = ModelMaker(light_dim, num_hidden_nodes)
    model.set_checkpoint_file(checkpoint_file)
    model.compile()
    model.load_weights()
    predictions = model.predict(test_data[start:end], target, batch_size) 
  
  






  






