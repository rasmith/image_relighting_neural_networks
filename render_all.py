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

import config
from cluster import *

def init(models_dir, img_dir):
  assert os.path.exists(models_dir)
  if not os.path.exists(img_dir):
    os.mkdir(img_dir) 
  return (models_dir, img_dir)

dirname = sys.argv[1]
image_number = int(sys.argv[2])

(model_dir, img_dir, width, height, num_images, ensemble_size,  max_levels, \
    sampled, assignments, average_img) = config.load_cfg(dirname)

init(model_dir, img_dir)

# Need to get [[L, i, i, i, i, i],
#              [L, i, i, i, i, i], ...
# to [x, y, j, r, g, b], ...
# load model file, execute
               
print("assignments.shape = %s, average_img.shape = %s"\
  % (assignments.shape, average_img.shape))

print("image_number = %s, num_images = %s" % (str(image_number), str(num_images)))

print("assignments.dtype= %s, average_img.dtype = %s"\
  % (str(assignments.dtype), str(average_img.dtype)))

# TODO - code this up               
test_data, ensemble_data = kmeans2d.assignment_data_to_test_data(\
  assignments, image_number, num_images, average_img)

print ("test_data.shape = %s, ensemble_data.shape = %s" % \
  (test_data.shape, ensemble_data.shape))

# void assignment_data_to_test_data(
    # int* assignment_data, int assignment_data_dim_1, int assignment_data_dim_2,
    # int assignment_data_dim_3, int image_number, int num_images,
    # float* average_image, int average_image_dim_1, int average_image_dim_2,
    # float** test_data, int* test_data_dim_1, int* test_data_dim_2,
    # int** ensemble_data, int* ensemble_data_dim_1, int* ensemble_data_dim_2);

# # predict image pixels
# for data in ensemble_data:
  # ensemble_id, level, start, end =  data
  # batch_size = start - end + 1
  # with tf.device('/cpu:0'):
    # model = ModelMaker(light_dim, num_hidden_nodes)
    # model.set_checkpoint_file(checkpoint_file)
    # model.compile()
    # model.load_weights()
    # predictions = model.predict(test_data[start:end], target, batch_size) 
  
  






  






