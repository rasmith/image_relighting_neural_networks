import sys
import time
import glob
import os
from scipy import misc
import numpy as np
import threading
from multiprocessing.dummy import Pool as ThreadPool

from cluster import *
from model import *

width = 696
height = 464
num_images = 1024
num_levels = 4
num_hidden_nodes = 15 
light_dim = 1
level = 0

max_clusters =  int(\
    maximum_clusters(width, height, num_hidden_nodes, num_images))

directory = sys.argv[1]

pixel_clusters = PixelClusters(directory, num_levels, max_clusters, timed = True)

total_estimate = 0
for indices, centers, labels, training_data, train_labels, batch_sizes  \
    in reversed(pixel_clusters):
  num_samples = len(indices)
  starts = np.zeros(len(batch_sizes), dtype=np.int32)
  ends = np.zeros(len(batch_sizes), dtype=np.int32)
  for i in range(0, len(batch_sizes)):
    if i > 0:
      starts[i] = batch_sizes[i] * num_samples + starts[i-1]
  for i in range(0, len(batch_sizes)):
    ends[i] = starts[i] + batch_sizes[i] * num_samples
  # get number of clusters
  num_clusters = len(centers)
  print("num_centers = %d\n" % len(centers))
  print ("len(batch_sizes) = %d\n" % len(batch_sizes))
  for cluster_id in range(0, 1):#len(centers)):
    start = time.time()
    model = ModelMaker(light_dim, num_hidden_nodes)
    model.set_checkpoint_file('model_'+str(level)+'-'+str(cluster_id)+'.hdf5')
    model.compile()
    model.train(training_data[starts[cluster_id]:ends[cluster_id], :], \
        train_labels[starts[cluster_id]:ends[cluster_id], :], 
        batch_sizes[cluster_id])
    end = time.time();
    print("time to train %f\n" % (end - start))
  total_estimate = num_clusters * (end - start) + total_estimate
  level = level + 1

print ("Estimate = %f\n" % (total_estimate))
