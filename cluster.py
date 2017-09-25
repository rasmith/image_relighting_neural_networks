import numpy as np
import time

import kmeans2d

class PixelCluster:
  def __init__(self, cluster_id, center, pixels):
    self.cluster_id = cluster_id
    self.center = center
    self.pixels = pixels

class PixelClusters:
  def __init__(self, max_clusters, extents):
    self.max_clusters = max_clusters
    self.levels = []
    self.extents = extents

  def cluster(self):
    width, height = self.extents
    pixels = [[x, y] for x in range(0, width) \
        for y in range(0, height)]
    pixels = np.array(pixels, dtype= float)
    num_clusters = self.max_clusters
    while num_clusters > 0:
      start = time.time()
      print("level = %d, num_clusters = %d" % (len(self.levels), num_clusters))
      cxx_centroids = kmeans2d.VectorFloat(2 * num_clusters)
      cxx_labels = kmeans2d.VectorInt(width * height)
      kmeans2d.kmeans2d(width, height, cxx_centroids, cxx_labels)
      centroids = [[cxx_centroids[i], cxx_centroids[i+1]] \
          for i in range(0, num_clusters)]
      labels = [cxx_labels[i] for i in range(0, width * height)]
      end = time.time()
      num_clusters = int(num_clusters / 4)
      self.levels.append([[centroids], [labels]])
      print("elapsed = %s" % str(end-start))

def pixels_required(num_weights, num_images):
  return 25 * num_images / num_images

def weights_required(num_hidden_nodes):
  return 12 * num_hidden_nodes + num_hidden_nodes * num_hidden_nodes + 3

def maximum_clusters(width, height, num_hidden_nodes, num_images):
  num_pixels = width * height
  num_weights = weights_required(num_hidden_nodes)
  num_pixels_needed = pixels_required(num_weights, num_images)
  return num_pixels / num_pixels_needed

max_num_clusters =  int(maximum_clusters(640, 480, 10, 1024))
print("maximum_clusters = %d" % max_num_clusters)
pc = PixelClusters(max_num_clusters, [640, 480])
pc.cluster()
