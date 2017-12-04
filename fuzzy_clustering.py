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

num_hidden_nodes = 15 
light_dim = 1

def train_model(level, images, cluster_id, cluster_to_pixels, pixel_counts):
    model = ModelMaker(light_dim, num_hidden_nodes)
    model.set_checkpoint_file('model_'+str(level)+'-'+str(cluster_id)+'.hdf5')
    model.compile()
    print("level = %d, cluster_id = %d" % (level, cluster_id))
    model.train(train_data, train_labels, batch_size)
    end = time.time();
    t = end - start
    print("[%d] %d - time to train total " % (level, cluster_id, t))

directory = sys.argv[1]

for centers, labels, training_data, train_labels, batch_sizes  \
    in reversed(pixel_clusters):
  # get number of clusters
  num_clusters = len(centers)
  print("num_centers = %d" % len(centers))


