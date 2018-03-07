from scipy import misc
import numpy as np
import matplotlib.image as mpimg

def save_cfg(dirname, average, sampled, assignments):
  cfg = dirname + '/cfg/relighting.cfg'
  height, width, assignment_size = assignments.shape
  ensemble_size = assignment_size - 1
  print("save_cfg: height = %d, width = %d, assignment_size = %d\n" %
      (height, width, assignment_size))
  print ("save_cfg:file = %s" % (cfg))
  # Open file.
  with open(cfg, "w") as f:
    f.write("%d\n" % (width)) # write width
    f.write("%d\n" % (height)) # write height
    f.write("%d\n" % (num_images)) # write num_images
    f.write("%d\n" % (ensemble_size)) # ensemble_size
    f.write("%s\n" % (' '.join([str(i) for i in sampled]))) # sampled images
    for x in range(0, width):
      for y in range(0, height):
          for i in range(0, assignment_size):
            f.write("%d" % assignments[y,x,i])
            if i < assignment_size - 1:
              f.write(" ")  
          f.write("\n")
  mpimg.imsave(dirname + '/average.png', average)

def load_cfg(dirname):
  cfg = dirname + '/cfg/relighting.cfg'
  with open(cfg, "r") as f:
    lines = f.readlines()
    width = int(lines[0])
    height = int(lines[1])
    num_images = int(lines[2])
    ensemble_size = int(lines[3])
    sampled = [int(i) for i in lines[4].split(' ')]
    assignment_size = ensemble_size + 1
    assignments = np.zeros((height, width, assignment_size))
    j = 5
    for y in range(0, height):
     for x in range(0, width):
        values = np.array(lines[j].split(" ")).astype(np.int)
        for i in range(0, assignment_size):
          assignments[y, x, i] = int(values[i])
        j = j + 1
  img_dir = dirname + '/img'
  model_dir= dirname + '/models'
  return model_dir, img_dir, width, height, num_images, sampled, assignments

