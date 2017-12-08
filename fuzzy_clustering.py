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
  for cluster_id in range(0, len(centers)):
    checkpoint_file = 'models/model_'+str(level)+'-'+str(cluster_id)+'.hdf5'
    print("[%d] %d/%d checkpoint_file = %s" %
          (level, cluster_id, len(centers) - 1, checkpoint_file))
    if not os.path.exists(checkpoint_file):
      continue
      start = time.time()
      with tf.device('/cpu:0'):
        config = tf.ConfigProto(intra_op_parallelism_threads=48,
                            inter_op_parallelism_threads=48, 
                            allow_soft_placement=True)    
        session = tf.Session(config=config)
        K.set_session(session)
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
    else:
      with tf.device('/cpu:0'):
        print("cluster_id = %d, start = %d, end = %d\n" %\
          (cluster_id, starts[cluster_id], ends[cluster_id]))
        model = ModelMaker(light_dim, num_hidden_nodes)
        model.set_checkpoint_file(checkpoint_file)
        model.compile()
        model.load_weights()
        print("sample_data = %s\n" % \
          train_data[starts[cluster_id]:starts[cluster_id]+10])
        print("sample_labels = %s\n" % \
          train_labels[starts[cluster_id]:starts[cluster_id]+10])
        score = model.test(train_data[starts[cluster_id]:ends[cluster_id], :],\
          train_labels[starts[cluster_id]:ends[cluster_id], :],\
          batch_sizes[cluster_id])
        print("\nscore=%3.8f, %3.8f\n" % (score[0], score[1]))

  level = level + 1

print ("Estimate = %f\n" % (total_estimate))
