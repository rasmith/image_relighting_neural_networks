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

from cluster import *

width = 696
height = 464
num_images = 1024
num_levels = 4
num_hidden_nodes = 15 
light_dim = 1
level = 0
ensemble_size = 5

max_clusters =  int(\
    maximum_clusters(width, height, num_hidden_nodes, num_images))

dirname = sys.argv[1]
timed = True

pixel_clusters = \
    PixelClusters(dirname, num_levels, max_clusters, ensemble_size, timed)

flagged_pixels = np.ones((width, height), dtype  = bool)
total_estimate = 0
for indices, centers, labels, closest, train_data, train_labels, batch_sizes\
    in reversed(pixel_clusters):
  print("closest.shape = %s\n" % str(closest.shape))
  num_samples = len(indices)
  starts = np.zeros(len(batch_sizes), dtype=np.int32)
  ends = np.zeros(len(batch_sizes), dtype=np.int32)
  for i in range(0, len(batch_sizes)):
    if i > 0:
      starts[i] = batch_sizes[i] * num_samples + starts[i-1]
  ends = starts + np.array(batch_sizes) * num_samples
  # get number of clusters
  num_clusters = len(centers)
  print("num_centers = %d\n" % len(centers))
  print ("len(batch_sizes) = %d\n" % len(batch_sizes))
  cluster_ids = range(0, len(centers))
  for cluster_id in cluster_ids:
    checkpoint_file = 'models/model_'+str(level)+'-'+str(cluster_id)+'.hdf5'
    print("[%d] %d/%d checkpoint_file = %s" %
          (level, cluster_id, len(centers) - 1, checkpoint_file))
    if not os.path.exists(checkpoint_file):
      start = time.time()
      with tf.device('/cpu:0'):
        model = ModelMaker(light_dim, num_hidden_nodes)
        model.set_checkpoint_file(checkpoint_file)
        model.compile()
        model.train(train_data[starts[cluster_id]:ends[cluster_id], :], \
            train_labels[starts[cluster_id]:ends[cluster_id], :], 
            batch_sizes[cluster_id])
        K.clear_session()
      end = time.time();
      print("[%d] %d/%d time to train %f\n" % \
          (level, cluster_id, len(centers), end - start))

  errors = np.ndarray((width, height), dtype = 'float', order='C')
  
  for cluster_id in cluster_ids:
    batch_size = batch_sizes[cluster_id]
    for k in range(0, ensemble_size):
      test, target = kmeans2d.closest_test_target(k, cluster_id, closest,\
                                                  train_data, target_data) 
      checkpoint_file = 'models/model_'+str(level)+'-'+str(cluster_id)+'.hdf5'
      print("[%d] %d/%d checkpoint_file = %s" %
            (level, cluster_id, len(centers) - 1, checkpoint_file))
      with tf.device('/cpu:0'):
        model = ModelMaker(light_dim, num_hidden_nodes)
        model.set_checkpoint_file(checkpoint_file)
        model.compile()
        model.load_weights()
        predictions = model.predict(test_data, batch_size) 
        kmeans2d.update_errors(test_data, target_data, predictions, errors)

  level = level + 1

print ("Estimate = %f\n" % (total_estimate))
