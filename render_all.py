import sys
import time
import glob
import os
from scipy import misc
import numpy as np
from multiprocessing import Pool
from model import ModelMaker
import tensorflow as tf
from keras import backend as K
import matplotlib.image as mpimg

from cluster import *

dirname = sys.argv[1]
if len(sys.argv) > 2:
  destdir = sys.argv[2]
else:
  destdir = ''


