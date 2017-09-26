import numpy as np
import time
import colorsys

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

      # # plot
      # palette = generate_palette(num_clusters)
      # image = render_clusters(width, height, labels, palette)
      # image = np.array(image)
      # image = image.reshape((width, height, 3))
      # plt.figure()
      # image_plot = plt.imshow(image)
      # plt.show(image_plot)

      self.levels.append([[centroids], [labels]])
      elapsed = float(int((end - start)*100))/100.0
      print("elapsed = %s" % str(elapsed))
      num_clusters = int(num_clusters / 4)

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

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  max_num_clusters =  int(maximum_clusters(640, 480, 10, 1024))
  print("maximum_clusters = %d" % max_num_clusters)
  width = 640
  height = 480
  channels = 3
  pc = PixelClusters(max_num_clusters, [width, height])
  pc.cluster()

  for i in range(0, len(pc.levels)):
    centers, labels = pc.levels[i]
    palette = generate_palette(len(centers[0]))
    print("num_centers = %d" % len(centers[0]))
    image = render_clusters(width, height, labels[0], palette)
    image = np.array(image)
    image = image.reshape((width, height, channels))
    plt.figure()
    image_plot = plt.imshow(image)
    plt.show(image_plot)
