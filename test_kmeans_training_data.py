import kmeans2d
import time
import numpy as np
import ctypes

# dirname = "/Users/randallsmith/image_relighting_neural_networks/data/bull/rgb"
dirname = "/home/agrippa/projects/image_relighting_neural_networks/data/bull/rgb"
num_centers = int(1200)
width = int(-1)
height = int(-1)
cxx_indices=kmeans2d.VectorInt()
cxx_centroids=kmeans2d.VectorFloat()
cxx_labels=kmeans2d.VectorInt()
cxx_batch_sizes = kmeans2d.VectorInt()
print("Running kmeans and getting training data and labels.\n")
start = time.time()
train_data = []
train_labels = []
width, height, train_data, train_labels\
= kmeans2d.kmeans_training_data(dirname, num_centers, cxx_indices, \
    cxx_centroids, cxx_labels, cxx_batch_sizes)
end = time.time()
print("Time: %f\n" % (end - start))
print ("width = %d, height = %d\n" % (width, height))
print ("indices.size = %d,  centroids.size = %d, labels.size = %d, batch_sizes.size = %d\n" \
    % (cxx_indices.size(), cxx_centroids.size(), cxx_labels.size(), cxx_batch_sizes.size()))
 
centroids = [[cxx_centroids[2*i], cxx_centroids[2*i+1]] for i in range(0,
  int(cxx_centroids.size() / 2))]
indices = [cxx_indices[i] for i in range(0, cxx_indices.size())]
labels = [cxx_labels[i] for i in range(0, cxx_labels.size())]
batch_sizes = [cxx_batch_sizes[i] for i in range(0, cxx_batch_sizes.size())]
# print("batch_sizes = %s" % str(batch_sizes))

print ("centroids.size = %d, labels.size = %d, batch_sizes.size = %d\n" \
    % (len(centroids), len(labels), len(batch_sizes)))
print ("train_data.size = %s, train_labels.size = %s\n" \
    % (str(train_data.shape), str(train_labels.shape)))

print("Getting k nearest neighbors.")
start = time.time()
k = int(5)
knns = kmeans2d.closest_n(width, height, k, cxx_centroids)
end = time.time()
# print("knns=%s\n" % knns)
print("Time: %f\n" % (end - start))
