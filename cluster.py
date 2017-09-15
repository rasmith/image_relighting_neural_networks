import numpy as np
import time

from scipy.cluster.vq import kmeans

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
      print("level = %d" % len(self.levels))
      centroids = kmeans(obs = pixels, k_or_guess = num_clusters, thresh = 1e-5)
      end = time.time()
      num_clusters = int(num_clusters / 4)
      self.levels.append([centroids])
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

print("maximum_clusters = %d" % maximum_clusters(640, 480, 10, 1024))
pc = PixelClusters(16, [640, 480])
pc.cluster()
