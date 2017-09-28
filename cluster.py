import numpy as np
import time
import colorsys
from itertools import takewhile, count

import kmeans2d

class PixelClusters:
  def __init__(self, num_levels, max_clusters, extents, timed = False):
    self.max_clusters = max_clusters
    self.levels = []
    self.extents = extents
    self.counts = list(takewhile(lambda x : x > 0, \
        map(lambda i: int(max_clusters/(4**i)), count(0, 1))))[0:num_levels]
    self.max_iteration = len(self.counts)
    self.iteration = 0
    self.timed = timed

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
    width, height = self.extents
    num_clusters = self.counts[self.iteration]
    cxx_centroids = kmeans2d.VectorFloat(2 * num_clusters)
    cxx_labels = kmeans2d.VectorInt(width * height)
    if self.timed:
      start = time.time()
    kmeans2d.kmeans2d(width, height, cxx_centroids, cxx_labels)
    if self.timed:
      end = time.time()
      elapsed = float(int((end - start)*100))/100.0
      print("iteration = %d elapsed = %s" % (self.iteration, str(elapsed)))
    centroids = [[cxx_centroids[i], cxx_centroids[i+1]] \
        for i in range(0, num_clusters)]
    labels = [cxx_labels[i] for i in range(0, width * height)]
    self.iteration += 1
    return (centroids, labels)

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
