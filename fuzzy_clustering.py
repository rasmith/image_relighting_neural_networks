import sys
import time
import glob
import os
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import threading
from multiprocessing.dummy import Pool as ThreadPool

from cluster import *
from model import *

width = 696
height = 464
num_images = 1024
num_levels = -1 
num_hidden_nodes = 15 
light_dim = 1
level = 0

max_clusters =  int(\
    maximum_clusters(width, height, num_hidden_nodes, num_images))

directory = sys.argv[1]

pixel_clusters = PixelClusters(directory, num_levels, max_clusters, timed = True)

for centers, labels, training_data, train_labels, batch_sizes  \
    in reversed(pixel_clusters):
  # get number of clusters
  num_clusters = len(centers)
  print("num_centers = %d\n" % len(centers))
  print ("len(batch_sizes) = %d\n" % len(batch_sizes))
  for cluster_id in range(0, len(centers)):
    start = time.time()
    model = ModelMaker(light_dim, num_hidden_nodes)
    model.set_checkpoint_file('model_'+str(level)+'-'+str(cluster_id)+'.hdf5')
    model.compile()
    model.train(training_data, train_labels, batch_sizes[cluster_id])
    end = time.time();
    print("time to train %f\n" % (end - start))
  level = level + 1


