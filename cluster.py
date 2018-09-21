import numpy as np
import time
import colorsys
import math
from itertools import takewhile, count

import kmeans2d

class PixelClusters:
  def __init__(self, directory, num_levels, max_clusters, ensemble_size, \
      timed = False):
    self.directory = directory
    self.max_clusters = max_clusters
    self.levels = []
    self.counts = list(takewhile(lambda x : x > 0, \
        map(lambda i: int(max_clusters/(4**i)), count(0, 1))))
    if num_levels != -1:
      self.counts = self.counts[0:num_levels]
    self.max_iteration = len(self.counts)
    self.iteration = 0
    self.timed = timed
    self.cxx_indices = kmeans2d.VectorInt()
    self.cxx_order  = kmeans2d.VectorInt()
    self.ensemble_size = ensemble_size

  def __iter__(self):
    return self

  def __reversed__(self):
    self.counts.reverse()
    return self

  def __next__(self):
    return self.next()

  def next(self):
    if self.iteration >= self.max_iteration:
      raise StopIteration
    width = int(-1)
    height = int(-1)
    num_centers = self.counts[self.iteration]
    cxx_centroids = kmeans2d.VectorFloat()
    cxx_labels = kmeans2d.VectorInt()
    cxx_batch_sizes = kmeans2d.VectorInt()
    if self.timed:
      start = time.time()
    k = self.ensemble_size
    width, height, train_data, train_labels, closest, average = \
      kmeans2d.kmeans_training_data(self.directory, num_centers, k, \
              self.cxx_indices, self.cxx_order, cxx_centroids, \
              cxx_labels, cxx_batch_sizes 
    print("len(cxx_indices) = %d\n" % self.cxx_indices.size())
    print("len(cxx_order) = %d\n" % self.cxx_order.size())
    print("len(cxx_batch_sizes) = %d\n" % cxx_batch_sizes.size())
    print("len(cxx_centroids) = %d\n" % cxx_centroids.size())
    print("len(cxx_labels) = %d\n" % cxx_labels.size())
    if self.timed:
      end = time.time()
      elapsed = float(int((end - start)*100))/100.0
      print("iteration = %d elapsed = %s" % (self.iteration, str(elapsed)))
    centroids = [[cxx_centroids[i], cxx_centroids[i+1]] \
        for i in range(0, num_centers)]
    cxx_centroids.clear()
    indices = [self.cxx_indices[i] for i in range(0, self.cxx_indices.size())]
    labels = [cxx_labels[i] for i in range(0, cxx_labels.size())]
    cxx_labels.clear()
    batch_sizes = [cxx_batch_sizes[i] for i in range(0, cxx_batch_sizes.size())]
    cxx_batch_sizes.clear()
    self.iteration += 1
    return (indices, self.cxx_order, centroids, labels, closest, average, \
            train_data, train_labels, batch_sizes)

def pixels_required(num_weights, num_images):
  return 25 * num_images / num_images

def weights_required(num_hidden_nodes):
  return 12 * num_hidden_nodes + num_hidden_nodes * num_hidden_nodes + 3

def maximum_clusters(width, height, num_hidden_nodes, num_images):
  num_pixels = width * height
  num_weights = weights_required(num_hidden_nodes)
  num_pixels_needed = pixels_required(num_weights, num_images)
  return num_pixels / num_pixels_needed

def generate_palette(num_centers):
  v = 0.5
  s = 0.5
  min_h = 0.0
  max_h = 1.0
  step = (max_h - min_h) / num_centers
  h = 0.0
  colors = [colorsys.hsv_to_rgb(step * float(i), s, v) for \
      i in range(0, num_centers)]
  return colors

def render_clusters(width, height, labels, palette):
  image = [palette[labels[width * y + x]]  \
      for x in range(0, width) for y in range(0, height)]
  return image
